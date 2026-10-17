package com.e7.bphelper

class CandidateRanker(
    private val metadata: EncoderMetadata,
) {
    fun rankAvailableRecommendations(
        scores: FloatArray,
        unavailableHeroes: Set<String>,
        candidateMask: FloatArray,
        topK: Int,
    ): RecommendationResult {
        val masked = scores.copyOf()
        for (index in masked.indices) {
            if (index < candidateMask.size && candidateMask[index] <= 0f) {
                masked[index] = 0f
            }
        }

        unavailableHeroes.forEach { heroCode ->
            val heroId = metadata.heroToId[heroCode]
            if (heroId != null && heroId in masked.indices) {
                masked[heroId] = 0f
            }
        }

        for (heroId in masked.indices) {
            val heroCode = metadata.idToHero.getOrNull(heroId).orEmpty()
            if (heroCode in metadata.maskSpecialHeroes) {
                masked[heroId] = 0f
            }
        }

        val total = masked.sum().toDouble()
        if (total <= 0.0) {
            return RecommendationResult(emptyList())
        }

        val normalized = masked.map { value -> value / total.toFloat() }
        val rankedIndices = normalized.indices
            .filter { index -> normalized[index] > 0f }
            .sortedByDescending { index -> normalized[index] }
            .take(topK)

        val heroes = rankedIndices.map { index ->
            RankedHero(
                heroCode = metadata.idToHero[index],
                rate = normalized[index] * 100.0,
            )
        }
        return RecommendationResult(topHeroes = heroes)
    }

    fun maskAndNormalizeProbs(
        probs: FloatArray,
        candidateMask: FloatArray,
    ): FloatArray {
        val masked = probs.copyOf()
        for (index in masked.indices) {
            if (index < candidateMask.size && candidateMask[index] <= 0f) {
                masked[index] = 0f
            }
        }
        val total = masked.sum()
        if (total <= 0f) {
            return masked
        }
        return FloatArray(masked.size) { index -> masked[index] / total }
    }

    fun blendConcreteRuleProbabilities(
        probsByRule: Map<String, FloatArray>,
    ): FloatArray {
        require(probsByRule.isNotEmpty()) { "Cannot blend warfare rule probabilities without concrete rule outputs" }
        val numHeroes = probsByRule.values.first().size
        val blended = FloatArray(numHeroes)
        var weightSum = 0.0
        metadata.warfareRuleOrder.forEach { rule ->
            val probs = probsByRule[rule] ?: return@forEach
            val weight = metadata.warfareRulePriors[rule] ?: 0.0
            if (weight <= 0.0) {
                return@forEach
            }
            for (index in 0 until numHeroes) {
                blended[index] += (weight * probs[index]).toFloat()
            }
            weightSum += weight
        }
        if (weightSum <= 0.0) {
            var count = 0
            metadata.warfareRuleOrder.forEach { rule ->
                val probs = probsByRule[rule] ?: return@forEach
                for (index in 0 until numHeroes) {
                    blended[index] += probs[index]
                }
                count += 1
            }
            if (count == 0) {
                return blended
            }
            return FloatArray(numHeroes) { index -> blended[index] / count.toFloat() }
        }
        return FloatArray(numHeroes) { index -> blended[index] / weightSum.toFloat() }
    }
}
