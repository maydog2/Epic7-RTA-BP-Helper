package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject

class PrebanRecommender(
    private val artifacts: RecommendationArtifacts,
) {
    fun recommend(
        excludedHeroes: List<String>,
        topK: Int = 10,
        source: String = "combined",
        firstPickSide: String? = null,
    ): JSONObject {
        val stats = artifacts.prebanStats
        val (counts, reason, usedFallback) = pickCounter(
            stats = stats,
            firstPickSide = firstPickSide,
            source = source,
        )

        val excluded = excludedHeroes.toSet()
        val totalMentions = counts.values.sum()
        val recommendations = JSONArray()

        if (totalMentions > 0) {
            counts.entries
                .sortedByDescending { (_, count) -> count }
                .forEach { (heroId, count) ->
                    if (heroId in excluded) {
                        return@forEach
                    }
                    val share = count.toDouble() / totalMentions.toDouble()
                    recommendations.put(
                        JSONObject()
                            .put("hero_id", heroId)
                            .put("normalized_preban_rate", share)
                            .put("preban_count", count)
                            .put("reason", reason),
                    )
                    if (recommendations.length() >= topK) {
                        return@forEach
                    }
                }
        }

        val topHeroes = JSONArray()
        val topRates = JSONArray()
        for (index in 0 until recommendations.length()) {
            val item = recommendations.getJSONObject(index)
            topHeroes.put(item.getString("hero_id"))
            topRates.put(item.getDouble("normalized_preban_rate") * 100.0)
        }

        return JSONObject()
            .put("phase", "preban")
            .put("preban_source", source)
            .put("first_pick_side", firstPickSide)
            .put("preban_stats_fallback", usedFallback)
            .put("recommendations", recommendations)
            .put("top_10_heroes", topHeroes)
            .put("top_10_rates", topRates)
    }

    fun parseTopK(rawValue: String?, default: Int = 10, maximum: Int = 50): Int {
        if (rawValue.isNullOrBlank()) {
            return default
        }
        val value = rawValue.toIntOrNull() ?: return default
        return value.coerceIn(1, maximum)
    }

    fun resolvePrebanSource(rawSide: String?): String {
        return when (rawSide?.trim()?.lowercase()) {
            "user" -> "ally"
            "enemy" -> "enemy"
            else -> "combined"
        }
    }

    fun resolvePrebanFirstPickSide(
        userFirstPickSide: String?,
        @Suppress("UNUSED_PARAMETER") prebanSource: String,
    ): String? {
        if (userFirstPickSide != DraftPositionUtils.FIRST_PICK_SIDE_ALLY &&
            userFirstPickSide != DraftPositionUtils.FIRST_PICK_SIDE_ENEMY
        ) {
            return null
        }
        return userFirstPickSide
    }

    private fun pickCounter(
        stats: PrebanStatsPayload,
        firstPickSide: String?,
        source: String,
    ): CounterSelection {
        val reason = when (source) {
            "ally" -> PREBAN_REASON_ALLY
            "enemy" -> PREBAN_REASON_ENEMY
            else -> PREBAN_REASON_COMBINED
        }
        val banSide = when (source) {
            "ally" -> "ally"
            "enemy" -> "enemy"
            else -> null
        }

        if (firstPickSide == DraftPositionUtils.FIRST_PICK_SIDE_ALLY ||
            firstPickSide == DraftPositionUtils.FIRST_PICK_SIDE_ENEMY
        ) {
            val context = stats.byContext[firstPickSide].orEmpty()
            val counts = if (banSide != null) {
                context[banSide].orEmpty()
            } else {
                mergeCounters(context["ally"].orEmpty(), context["enemy"].orEmpty())
            }
            if (counts.values.sum() > 0) {
                return CounterSelection(counts, reason, usedFallback = false)
            }
        }

        val fallbackCounts = if (banSide != null) {
            stats.combined[banSide].orEmpty()
        } else {
            mergeCounters(stats.combined["ally"].orEmpty(), stats.combined["enemy"].orEmpty())
        }
        return CounterSelection(fallbackCounts, reason, usedFallback = true)
    }

    private fun mergeCounters(
        left: Map<String, Int>,
        right: Map<String, Int>,
    ): Map<String, Int> {
        val merged = left.toMutableMap()
        right.forEach { (hero, count) ->
            merged[hero] = merged.getOrDefault(hero, 0) + count
        }
        return merged
    }

    private data class CounterSelection(
        val counts: Map<String, Int>,
        val reason: String,
        val usedFallback: Boolean,
    )

    companion object {
        private const val PREBAN_REASON_COMBINED = "High preban frequency in historical RTA matches."
        private const val PREBAN_REASON_ALLY = "Often prebanned on your side in historical RTA matches."
        private const val PREBAN_REASON_ENEMY = "Often prebanned on the opponent side in historical RTA matches."
    }
}
