package com.e7.bphelper

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TfliteRecommendEngine(
    context: Context,
    private val metadata: EncoderMetadata,
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val interpreter: Interpreter = Interpreter(loadModelFile())
    private val arrayBuilder = InferenceArrayBuilder(metadata)
    private val ranker = CandidateRanker(metadata)
    private val inputIndexByName: Map<String, Int> = buildInputIndexMap()

    fun predict(input: DraftRecommendInput): RecommendationResult {
        val tensors = arrayBuilder.build(input)
        return predictWithScores(input, tensors).toRecommendationResult()
    }

    fun predictWithScores(
        input: DraftRecommendInput,
        tensors: InferenceTensors,
    ): TflitePickResult {
        val normalizedRule = normalizeWarfareRules(input.warfareRules)
        val probs = if (normalizedRule == InferenceArrayBuilder.WARFARE_RULE_ANY) {
            predictAnyRule(tensors)
        } else {
            val ruleTensors = tensors.withWarfareRuleId(metadata.warfareRuleId(normalizedRule))
            ranker.maskAndNormalizeProbs(runInference(ruleTensors), tensors.candidateMask)
        }

        val unavailableHeroes = unavailableHeroesFromDraft(input)
        val ranked = ranker.rankAvailableRecommendations(
            scores = probs,
            unavailableHeroes = unavailableHeroes,
            candidateMask = tensors.candidateMask,
            topK = input.topK,
        )
        val modelScores = ranked.topHeroes.associate { hero ->
            val heroId = metadata.heroToId[hero.heroCode]
            val rawScore = if (heroId != null && heroId in probs.indices) {
                probs[heroId].toDouble()
            } else {
                0.0
            }
            hero.heroCode to rawScore
        }
        return TflitePickResult(
            topHeroes = ranked.topHeroes,
            modelScores = modelScores,
        )
    }

    private fun predictAnyRule(tensors: InferenceTensors): FloatArray {
        val probsByRule = linkedMapOf<String, FloatArray>()
        metadata.warfareRuleOrder.forEach { rule ->
            val ruleTensors = tensors.withWarfareRuleId(metadata.warfareRuleId(rule))
            val raw = runInference(ruleTensors)
            probsByRule[rule] = ranker.maskAndNormalizeProbs(raw, tensors.candidateMask)
        }
        val blended = ranker.blendConcreteRuleProbabilities(probsByRule)
        return ranker.maskAndNormalizeProbs(blended, tensors.candidateMask)
    }

    private fun runInference(tensors: InferenceTensors): FloatArray {
        val inputs = buildNamedInputs(tensors)
        val outputBuffer = Array(1) { FloatArray(metadata.numHeroes) }
        val inputBuffers = orderInputs(inputs)
        interpreter.runForMultipleInputsOutputs(inputBuffers, mapOf(0 to outputBuffer))
        return outputBuffer[0]
    }

    private fun buildNamedInputs(tensors: InferenceTensors): Map<String, Any> {
        return mapOf(
            "draft_hero_ids" to intBatch(tensors.draftHeroIds),
            "draft_role_ids" to intBatch(tensors.draftRoleIds),
            "draft_element_ids" to intBatch(tensors.draftElementIds),
            "draft_side_ids" to intBatch(tensors.draftSideIds),
            "draft_order_ids" to intBatch(tensors.draftOrderIds),
            "draft_position_bucket_ids" to intBatch(tensors.draftPositionBucketIds),
            "is_first_pick_slot" to intBatch(tensors.isFirstPickSlot),
            "ally_preban_ids" to intBatch(tensors.allyPrebanIds),
            "ally_preban_role_ids" to intBatch(tensors.allyPrebanRoleIds),
            "ally_preban_element_ids" to intBatch(tensors.allyPrebanElementIds),
            "enemy_preban_ids" to intBatch(tensors.enemyPrebanIds),
            "enemy_preban_role_ids" to intBatch(tensors.enemyPrebanRoleIds),
            "enemy_preban_element_ids" to intBatch(tensors.enemyPrebanElementIds),
            "draft_filled_mask" to floatBatch(tensors.draftFilledMask),
            "candidate_mask" to floatBatch(tensors.candidateMask),
            "current_position_bucket_id" to intScalar2d(tensors.currentPositionBucketId),
            "current_order" to intScalar2d(tensors.currentOrder),
            "current_side_id" to intScalar2d(tensors.currentSideId),
            "warfare_rule_id" to intScalar2d(tensors.warfareRuleId),
        )
    }

    private fun orderInputs(inputs: Map<String, Any>): Array<Any> {
        val ordered = arrayOfNulls<Any>(interpreter.inputTensorCount)
        inputs.forEach { (name, value) ->
            val index = inputIndexByName[name]
                ?: throw IllegalStateException("TFLite model is missing input '$name'")
            ordered[index] = value
        }
        if (ordered.any { it == null }) {
            val missing = inputIndexByName.entries
                .filter { (_, index) -> ordered[index] == null }
                .map { it.key }
            throw IllegalStateException("Missing TFLite inputs: $missing")
        }
        @Suppress("UNCHECKED_CAST")
        return ordered as Array<Any>
    }

    private fun buildInputIndexMap(): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        for (index in 0 until interpreter.inputTensorCount) {
            val name = normalizeTensorName(interpreter.getInputTensor(index).name())
            map[name] = index
        }
        return map
    }

    private fun normalizeTensorName(name: String): String {
        var cleaned = name.substringBefore(':')
        if (cleaned.startsWith("serving_default_")) {
            cleaned = cleaned.removePrefix("serving_default_")
        }
        return cleaned
    }

    private fun intBatch(values: IntArray): Array<IntArray> = arrayOf(values)

    private fun floatBatch(values: FloatArray): Array<FloatArray> = arrayOf(values)

    private fun intScalar2d(value: Int): Array<IntArray> = arrayOf(intArrayOf(value))

    private fun unavailableHeroesFromDraft(input: DraftRecommendInput): Set<String> {
        return buildSet {
            input.userPicks.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.enemyPicks.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.allyPreban.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.enemyPreban.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
        }
    }

    private fun normalizeWarfareRules(rawValue: String?): String {
        val text = rawValue?.trim().orEmpty()
        if (text.isEmpty() || text.equals(InferenceArrayBuilder.WARFARE_RULE_ANY, ignoreCase = true)) {
            return InferenceArrayBuilder.WARFARE_RULE_ANY
        }
        require(text in metadata.warfareRuleOrder) {
            "Invalid warfare_rules value '$rawValue'"
        }
        return text
    }

    private fun loadModelFile(): ByteBuffer {
        val assetPath = metadata.tfliteAssetPath
        val assetManager = appContext.assets
        return if (assetPath.startsWith("data/")) {
            assetManager.openFd(assetPath).use { assetFd ->
                FileInputStream(assetFd.fileDescriptor).use { inputStream ->
                    inputStream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        assetFd.startOffset,
                        assetFd.declaredLength,
                    ).apply {
                        rewind()
                    }
                }
            }
        } else {
            val bytes = assetManager.open(assetPath).use { it.readBytes() }
            ByteBuffer.allocateDirect(bytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(bytes)
                rewind()
            }.asReadOnlyBuffer()
        }
    }

    override fun close() {
        interpreter.close()
    }
}
