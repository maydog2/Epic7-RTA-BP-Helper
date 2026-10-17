package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject

class FirstPickRecommender(
    private val artifacts: RecommendationArtifacts,
) {
    fun recommend(
        allyPreban: List<String>,
        enemyPreban: List<String>,
        firstPickSide: String,
        excludedHeroes: Set<String> = emptySet(),
        topK: Int = 10,
    ): JSONObject {
        if (firstPickSide != DraftPositionUtils.FIRST_PICK_SIDE_ALLY &&
            firstPickSide != DraftPositionUtils.FIRST_PICK_SIDE_ENEMY
        ) {
            return emptyPayload(fallbackLevel = 4)
        }

        val records = artifacts.firstPickRecords.filter { record ->
            record.firstPickSide == firstPickSide
        }
        val (queryFirst, querySecond) = deriveDirectionalPrebans(
            firstPickSide = firstPickSide,
            allyPreban = allyPreban,
            enemyPreban = enemyPreban,
        )

        val rankedRecommendations = mutableListOf<Pair<String, Double>>()
        val seenHeroes = mutableSetOf<String>()
        var firstUsedLevel: Int? = null
        var filledThroughLevel = 4

        for (level in listOf(1, 2, 3)) {
            val weightedCounts = mutableMapOf<String, Double>()
            records.forEach { record ->
                val matchLevel = directionalMatchLevel(
                    queryFirst = queryFirst,
                    querySecond = querySecond,
                    matchFirst = record.firstSidePreban,
                    matchSecond = record.secondSidePreban,
                ) ?: return@forEach
                if (matchLevel != level) {
                    return@forEach
                }
                val weight = if (level < 3) {
                    1.0
                } else {
                    directionalSimilarity(
                        queryFirst = queryFirst,
                        querySecond = querySecond,
                        matchFirst = record.firstSidePreban,
                        matchSecond = record.secondSidePreban,
                    )
                }
                weightedCounts[record.order1Hero] = weightedCounts.getOrDefault(record.order1Hero, 0.0) + weight
            }

            if (weightedCounts.isNotEmpty()) {
                if (firstUsedLevel == null) {
                    firstUsedLevel = level
                }
                appendRankedCounts(
                    recommendations = rankedRecommendations,
                    weightedCounts = weightedCounts,
                    excludedHeroes = excludedHeroes,
                    seenHeroes = seenHeroes,
                    fallbackLevel = level,
                )
                filledThroughLevel = level
            }
        }

        val seasonCounts = records.groupingBy { it.order1Hero }.eachCount()
            .mapValues { (_, count) -> count.toDouble() }
        if (firstUsedLevel == null && seasonCounts.isNotEmpty()) {
            firstUsedLevel = 4
        }
        appendRankedCounts(
            recommendations = rankedRecommendations,
            weightedCounts = seasonCounts,
            excludedHeroes = excludedHeroes,
            seenHeroes = seenHeroes,
            fallbackLevel = 4,
        )

        return formatRankedRecommendations(
            recommendations = rankedRecommendations,
            fallbackLevel = firstUsedLevel ?: 4,
            filledThroughLevel = filledThroughLevel,
            topK = topK,
        )
    }

    private fun normalizePrebanList(prebans: List<String>, maxSize: Int = 2): List<String> {
        return prebans.filter { it.isNotBlank() }.take(maxSize)
    }

    private fun deriveDirectionalPrebans(
        firstPickSide: String,
        allyPreban: List<String>,
        enemyPreban: List<String>,
    ): Pair<List<String>, List<String>> {
        val ally = normalizePrebanList(allyPreban)
        val enemy = normalizePrebanList(enemyPreban)
        return if (firstPickSide == DraftPositionUtils.FIRST_PICK_SIDE_ALLY) {
            ally to enemy
        } else {
            enemy to ally
        }
    }

    private fun prebanOverlap(query: List<String>, candidate: List<String>): Double {
        if (query.isEmpty() && candidate.isEmpty()) {
            return 1.0
        }
        if (query.isEmpty() || candidate.isEmpty()) {
            return 0.0
        }
        val overlap = query.toSet().intersect(candidate.toSet()).size
        return overlap.toDouble() / maxOf(query.size, candidate.size).toDouble()
    }

    private fun directionalMatchLevel(
        queryFirst: List<String>,
        querySecond: List<String>,
        matchFirst: List<String>,
        matchSecond: List<String>,
    ): Int? {
        val firstOverlap = prebanOverlap(queryFirst, matchFirst)
        val secondOverlap = prebanOverlap(querySecond, matchSecond)

        if (queryFirst == matchFirst && querySecond == matchSecond) {
            return 1
        }
        if (queryFirst == matchFirst && secondOverlap > 0.0 && querySecond != matchSecond) {
            return 2
        }
        if (firstOverlap > 0.0 && secondOverlap > 0.0) {
            return 3
        }
        return null
    }

    private fun directionalSimilarity(
        queryFirst: List<String>,
        querySecond: List<String>,
        matchFirst: List<String>,
        matchSecond: List<String>,
    ): Double {
        return 0.6 * prebanOverlap(queryFirst, matchFirst) +
            0.4 * prebanOverlap(querySecond, matchSecond)
    }

    private fun appendRankedCounts(
        recommendations: MutableList<Pair<String, Double>>,
        weightedCounts: Map<String, Double>,
        excludedHeroes: Set<String>,
        seenHeroes: MutableSet<String>,
        fallbackLevel: Int,
    ) {
        val levelWeight = FALLBACK_LEVEL_WEIGHT.getValue(fallbackLevel)
        val eligible = weightedCounts.entries
            .sortedByDescending { (_, weight) -> weight }
            .filter { (hero, _) -> hero !in excludedHeroes && hero !in seenHeroes }
        val levelTotal = eligible.sumOf { (_, weight) -> weight }
        if (levelTotal <= 0.0) {
            return
        }

        eligible.forEach { (hero, weight) ->
            recommendations += hero to (weight / levelTotal * levelWeight)
            seenHeroes += hero
        }
    }

    private fun formatRankedRecommendations(
        recommendations: List<Pair<String, Double>>,
        fallbackLevel: Int,
        filledThroughLevel: Int,
        topK: Int,
    ): JSONObject {
        val ranked = recommendations.sortedByDescending { (_, weight) -> weight }.take(topK)
        val total = ranked.sumOf { (_, weight) -> weight }
        if (total <= 0.0) {
            return emptyPayload(fallbackLevel = fallbackLevel, filledThroughLevel = filledThroughLevel)
        }

        val topHeroes = JSONArray()
        val topRates = JSONArray()
        ranked.forEach { (hero, weight) ->
            topHeroes.put(hero)
            topRates.put(weight / total * 100.0)
        }

        return JSONObject()
            .put("phase", "pick")
            .put("handled_by", "first_pick_stats")
            .put("first_pick_fallback_level", fallbackLevel)
            .put("first_pick_filled_through_level", filledThroughLevel)
            .put("reason", FIRST_PICK_REASON)
            .put("top_10_heroes", topHeroes)
            .put("top_10_rates", topRates)
    }

    private fun emptyPayload(
        fallbackLevel: Int,
        filledThroughLevel: Int = 4,
    ): JSONObject {
        return JSONObject()
            .put("phase", "pick")
            .put("handled_by", "first_pick_stats")
            .put("first_pick_fallback_level", fallbackLevel)
            .put("first_pick_filled_through_level", filledThroughLevel)
            .put("top_10_heroes", JSONArray())
            .put("top_10_rates", JSONArray())
    }

    companion object {
        private const val FIRST_PICK_REASON =
            "Historical order-1 first pick with similar directional preban context."
        private val FALLBACK_LEVEL_WEIGHT = mapOf(
            1 to 1.0,
            2 to 0.50,
            3 to 0.25,
            4 to 0.05,
        )
    }
}
