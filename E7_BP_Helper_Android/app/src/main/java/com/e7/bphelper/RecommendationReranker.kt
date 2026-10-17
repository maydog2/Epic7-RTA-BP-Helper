package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max

class RecommendationReranker(
    private val artifacts: RecommendationArtifacts,
) {
    private val synergyLookup: Map<Pair<String, String>, SynergyEntry> by lazy { buildSynergyLookup() }
    private val responseLookup: Map<Triple<String, String, String>, ResponseEntry> by lazy { buildResponseLookup() }
    private val openingResponseLookup: Map<Triple<String, String, String>, ResponseEntry> by lazy {
        buildOpeningResponseLookup()
    }

    fun rerankCandidates(
        candidates: List<String>,
        modelScores: Map<String, Double>,
        allyPicks: List<String>,
        enemyPicks: List<String>,
        unavailableHeroes: Set<String>,
        positionBucket: String,
        firstPickTeam: String?,
        topK: Int = 10,
    ): RerankResult {
        if (positionBucket == "1") {
            throw IllegalArgumentException("Bucket 1 must be handled by first_pick_recommender, not reranker")
        }

        val filteredCandidates = candidates
            .filter { candidate ->
                candidate.isNotBlank() &&
                    candidate !in unavailableHeroes &&
                    candidate != "unknown"
            }
            .take(topK)

        if (filteredCandidates.isEmpty()) {
            return RerankResult(
                topHeroes = emptyList(),
                topRates = emptyList(),
                usedReranker = false,
                positionBucket = positionBucket,
            )
        }

        val modelScoreNorm = normalizeModelScores(modelScores, filteredCandidates)
        val weights = bucketWeights(positionBucket)
        val firstPickSide = DraftPositionUtils.parseFirstPickSide(firstPickTeam)
        val componentScores = mutableMapOf<String, ComponentScores>()
        var hasAnyStatSignal = false

        filteredCandidates.forEach { candidate ->
            val synergyScore = lookupSynergy(candidate, allyPicks)
            val responseScore = lookupEnemyResponseScore(
                candidate = candidate,
                allyPicks = allyPicks,
                enemyPicks = enemyPicks,
                positionBucket = positionBucket,
                firstPickSide = firstPickSide,
            )
            if (synergyScore > 0.0 || responseScore > 0.0) {
                hasAnyStatSignal = true
            }
            val finalScore = weights.model * modelScoreNorm.getValue(candidate) +
                weights.synergy * synergyScore +
                weights.response * responseScore
            componentScores[candidate] = ComponentScores(
                modelScore = modelScoreNorm.getValue(candidate),
                allySynergyScore = synergyScore,
                enemyResponseScore = responseScore,
                finalScore = finalScore,
            )
        }

        if (!hasAnyStatSignal) {
            val ordered = applyLowPickTop3Guard(
                ordered = filteredCandidates,
                componentScores = componentScores,
            )
            val rates = scoreToPercentages(ordered, componentScores)
            return RerankResult(
                topHeroes = ordered,
                topRates = rates.map { round2(it) },
                usedReranker = false,
                positionBucket = positionBucket,
            )
        }

        val ordered = filteredCandidates.sortedWith(
            compareByDescending<String> { componentScores.getValue(it).finalScore }
                .thenByDescending { componentScores.getValue(it).modelScore },
        )
        val guarded = applyLowPickTop3Guard(
            ordered = ordered,
            componentScores = componentScores,
        )
        val rates = scoreToPercentages(guarded, componentScores)
        return RerankResult(
            topHeroes = guarded,
            topRates = rates.map { round2(it) },
            usedReranker = true,
            positionBucket = positionBucket,
        )
    }

    private fun lookupSynergy(candidate: String, allyPicks: List<String>): Double {
        val matchedScores = allyPicks.mapNotNull { allyHero ->
            if (allyHero.isBlank() || allyHero == "unknown" || allyHero == candidate) {
                return@mapNotNull null
            }
            synergyLookup[candidate to allyHero]?.score
        }
        return combineComponentScores(matchedScores, "synergy")
    }

    private fun lookupGeneralResponse(
        candidate: String,
        enemyPicks: List<String>,
        positionBucket: String,
    ): Double {
        val matchedScores = enemyPicks.mapNotNull { enemyHero ->
            if (enemyHero.isBlank() || enemyHero == "unknown" || enemyHero == candidate) {
                return@mapNotNull null
            }
            responseLookup[Triple(positionBucket, candidate, enemyHero)]?.score
        }
        return combineComponentScores(matchedScores, positionBucket)
    }

    private fun lookupEnemyResponseScore(
        candidate: String,
        allyPicks: List<String>,
        enemyPicks: List<String>,
        positionBucket: String,
        firstPickSide: String?,
    ): Double {
        if (positionBucket == "2_3" &&
            (firstPickSide == DraftPositionUtils.FIRST_PICK_SIDE_ALLY ||
                firstPickSide == DraftPositionUtils.FIRST_PICK_SIDE_ENEMY)
        ) {
            val enemyOrder1 = opposingOrder1Hero(firstPickSide, allyPicks, enemyPicks)
            if (!enemyOrder1.isNullOrBlank()) {
                openingResponseLookup[Triple(firstPickSide, enemyOrder1, candidate)]?.score?.let {
                    return it
                }
            }
        }
        return lookupGeneralResponse(candidate, enemyPicks, positionBucket)
    }

    private fun opposingOrder1Hero(
        firstPickSide: String?,
        allyPicks: List<String>,
        enemyPicks: List<String>,
    ): String? {
        return when (firstPickSide) {
            DraftPositionUtils.FIRST_PICK_SIDE_ENEMY -> enemyPicks.firstOrNull()
            DraftPositionUtils.FIRST_PICK_SIDE_ALLY -> allyPicks.firstOrNull()
            else -> null
        }
    }

    private fun combineComponentScores(scores: List<Double>, positionBucket: String): Double {
        if (scores.isEmpty()) {
            return 0.0
        }
        val average = scores.average()
        val maximum = scores.maxOrNull() ?: 0.0
        return if (positionBucket == "10") {
            max(average, maximum)
        } else {
            max(average, 0.7 * maximum)
        }
    }

    private fun normalizeModelScores(
        modelScores: Map<String, Double>,
        candidates: List<String>,
    ): Map<String, Double> {
        val values = candidates.map { candidate -> max(modelScores[candidate] ?: 0.0, 0.0) }
        val total = values.sum()
        if (total <= 0.0) {
            val uniform = if (candidates.isEmpty()) 0.0 else 1.0 / candidates.size.toDouble()
            return candidates.associateWith { uniform }
        }
        return candidates.zip(values).associate { (candidate, value) -> candidate to value / total }
    }

    private fun scoreToPercentages(
        orderedCandidates: List<String>,
        componentScores: Map<String, ComponentScores>,
    ): List<Double> {
        val total = orderedCandidates.sumOf { candidate ->
            max(componentScores[candidate]?.finalScore ?: 0.0, 0.0)
        }
        if (total <= 0.0) {
            return List(orderedCandidates.size) { 0.0 }
        }
        return orderedCandidates.map { candidate ->
            max(componentScores[candidate]?.finalScore ?: 0.0, 0.0) / total * 100.0
        }
    }

    private fun applyLowPickTop3Guard(
        ordered: List<String>,
        componentScores: Map<String, ComponentScores>,
    ): List<String> {
        if (!LOW_PICK_GUARD_ENABLED) {
            return ordered
        }

        val top3Blocked = mutableMapOf<String, Boolean>()
        ordered.forEach { candidate ->
            val statCount = artifacts.heroAppearanceCounts[candidate] ?: 0
            val scores = componentScores[candidate] ?: return@forEach
            val supportingEvidence = evaluateLowPickSupport(
                modelScoreNorm = scores.modelScore,
                allySynergyScore = scores.allySynergyScore,
                enemyResponseScore = scores.enemyResponseScore,
            )
            val isLowPick = statCount < LOW_PICK_COUNT_THRESHOLD
            top3Blocked[candidate] = isLowPick && !supportingEvidence.values.any { it }
        }
        return reorderWithLowPickTop3Guard(ordered, top3Blocked)
    }

    private fun evaluateLowPickSupport(
        modelScoreNorm: Double,
        allySynergyScore: Double,
        enemyResponseScore: Double,
    ): Map<String, Boolean> {
        return mapOf(
            "model_confident" to (modelScoreNorm >= LOW_PICK_MODEL_CONFIDENCE_THRESHOLD),
            "response_supported" to (enemyResponseScore >= LOW_PICK_RESPONSE_THRESHOLD),
            "synergy_supported" to (allySynergyScore >= LOW_PICK_SYNERGY_THRESHOLD),
        )
    }

    private fun reorderWithLowPickTop3Guard(
        ordered: List<String>,
        top3Blocked: Map<String, Boolean>,
    ): List<String> {
        val eligible = ordered.filter { candidate -> top3Blocked[candidate] != true }
        val blocked = ordered.filter { candidate -> top3Blocked[candidate] == true }
        val top3 = eligible.take(3).toMutableList()
        if (top3.size < 3) {
            top3 += blocked.take(3 - top3.size)
        }
        val used = top3.toSet()
        val rest = ordered.filter { candidate -> candidate !in used }
        return top3 + rest
    }

    private fun bucketWeights(positionBucket: String): BucketWeights {
        val weights = BUCKET_WEIGHTS[positionBucket] ?: DEFAULT_BUCKET_WEIGHTS
        return BucketWeights(
            model = weights.getValue("model"),
            synergy = weights.getValue("synergy"),
            response = weights.getValue("response"),
        )
    }

    private fun buildSynergyLookup(): Map<Pair<String, String>, SynergyEntry> {
        val lookup = mutableMapOf<Pair<String, String>, SynergyEntry>()
        artifacts.synergyStats.forEach { row ->
            if (row.sameTeamCount < MIN_SYNERGY_COUNT) {
                return@forEach
            }
            val entry = SynergyEntry(score = row.normalizedSynergyScore, sameTeamCount = row.sameTeamCount)
            lookup[row.heroA to row.heroB] = entry
            lookup[row.heroB to row.heroA] = entry
        }
        return lookup
    }

    private fun buildResponseLookup(): Map<Triple<String, String, String>, ResponseEntry> {
        val lookup = mutableMapOf<Triple<String, String, String>, ResponseEntry>()
        artifacts.responseStats.forEach { row ->
            if (row.responseCount < MIN_RESPONSE_COUNT) {
                return@forEach
            }
            lookup[Triple(row.positionBucket, row.candidateHero, row.enemyHero)] = ResponseEntry(
                score = row.normalizedResponseScore,
                responseCount = row.responseCount,
            )
        }
        return lookup
    }

    private fun buildOpeningResponseLookup(): Map<Triple<String, String, String>, ResponseEntry> {
        val lookup = mutableMapOf<Triple<String, String, String>, ResponseEntry>()
        artifacts.openingResponseStats.forEach { row ->
            if (row.responseCount < MIN_RESPONSE_COUNT) {
                return@forEach
            }
            lookup[Triple(row.firstPickSide, row.enemyOrder1Hero, row.candidateHero)] = ResponseEntry(
                score = row.normalizedResponseScore,
                responseCount = row.responseCount,
            )
        }
        return lookup
    }

    private fun round2(value: Double): Double {
        return "%.2f".format(value).toDouble()
    }

    data class RerankResult(
        val topHeroes: List<String>,
        val topRates: List<Double>,
        val usedReranker: Boolean,
        val positionBucket: String,
    )

    private data class ComponentScores(
        val modelScore: Double,
        val allySynergyScore: Double,
        val enemyResponseScore: Double,
        val finalScore: Double,
    )

    private data class SynergyEntry(
        val score: Double,
        val sameTeamCount: Int,
    )

    private data class ResponseEntry(
        val score: Double,
        val responseCount: Int,
    )

    private data class BucketWeights(
        val model: Double,
        val synergy: Double,
        val response: Double,
    )

    companion object {
        private const val MIN_SYNERGY_COUNT = 3
        private const val MIN_RESPONSE_COUNT = 3
        private const val LOW_PICK_COUNT_THRESHOLD = 35
        private const val LOW_PICK_MODEL_CONFIDENCE_THRESHOLD = 0.28
        private const val LOW_PICK_RESPONSE_THRESHOLD = 0.22
        private const val LOW_PICK_SYNERGY_THRESHOLD = 0.22
        private const val LOW_PICK_GUARD_ENABLED = true

        private val BUCKET_WEIGHTS = mapOf(
            "2_3" to mapOf("model" to 0.55, "synergy" to 0.15, "response" to 0.30),
            "4" to mapOf("model" to 0.65, "synergy" to 0.20, "response" to 0.15),
            "5_protected" to mapOf("model" to 0.55, "synergy" to 0.35, "response" to 0.10),
            "6_protected" to mapOf("model" to 0.55, "synergy" to 0.25, "response" to 0.20),
            "7" to mapOf("model" to 0.60, "synergy" to 0.15, "response" to 0.25),
            "8_9" to mapOf("model" to 0.50, "synergy" to 0.20, "response" to 0.30),
            "10" to mapOf("model" to 0.50, "synergy" to 0.15, "response" to 0.35),
        )
        private val DEFAULT_BUCKET_WEIGHTS = mapOf(
            "model" to 0.60,
            "synergy" to 0.20,
            "response" to 0.20,
        )
    }
}
