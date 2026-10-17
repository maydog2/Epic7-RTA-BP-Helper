package com.e7.bphelper

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class RecommendationArtifacts(context: Context) {
    private val appContext = context.applicationContext

    val prebanStats: PrebanStatsPayload by lazy { loadPrebanStats() }
    val firstPickRecords: List<FirstPickRecord> by lazy { loadFirstPickRecords() }
    val synergyStats: List<SynergyStatRow> by lazy { loadSynergyStats() }
    val responseStats: List<ResponseStatRow> by lazy { loadResponseStats() }
    val openingResponseStats: List<OpeningResponseStatRow> by lazy { loadOpeningResponseStats() }
    val finalBanConfig: FinalBanConfig by lazy { loadFinalBanConfig() }
    val finalBanStats: FinalBanStatsPayload by lazy { loadFinalBanStats() }
    val heroAppearanceCounts: Map<String, Int> by lazy { loadHeroAppearanceCounts() }

    private fun loadAssetText(path: String): String {
        return appContext.assets.open(path).use { stream ->
            stream.readBytes().toString(Charsets.UTF_8)
        }
    }

    private fun loadPrebanStats(): PrebanStatsPayload {
        val root = JSONObject(loadAssetText(PREBAN_STATS_PATH))
        return PrebanStatsPayload(
            byContext = parseCounterContext(root.getJSONObject("by_context")),
            combined = parseCounterSides(root.getJSONObject("combined")),
        )
    }

    private fun parseCounterContext(root: JSONObject): Map<String, Map<String, Map<String, Int>>> {
        val result = mutableMapOf<String, Map<String, Map<String, Int>>>()
        root.keys().forEach { fps ->
            result[fps] = parseCounterSides(root.getJSONObject(fps))
        }
        return result
    }

    private fun parseCounterSides(root: JSONObject): Map<String, Map<String, Int>> {
        val result = mutableMapOf<String, Map<String, Int>>()
        root.keys().forEach { side ->
            result[side] = parseCounter(root.getJSONObject(side))
        }
        return result
    }

    private fun parseCounter(root: JSONObject): Map<String, Int> {
        val result = mutableMapOf<String, Int>()
        root.keys().forEach { hero ->
            result[hero] = root.getInt(hero)
        }
        return result
    }

    private fun loadFirstPickRecords(): List<FirstPickRecord> {
        val array = JSONArray(loadAssetText(FIRST_PICK_RECORDS_PATH))
        return List(array.length()) { index ->
            val item = array.getJSONObject(index)
            FirstPickRecord(
                firstPickSide = item.getString("first_pick_side"),
                firstSidePreban = item.getStringList("first_side_preban"),
                secondSidePreban = item.getStringList("second_side_preban"),
                order1Hero = item.getString("order_1_hero"),
                season = item.optString("season").ifBlank { null },
            )
        }
    }

    private fun loadSynergyStats(): List<SynergyStatRow> {
        return loadStatArray(SYNERGY_STATS_PATH) { item ->
            SynergyStatRow(
                heroA = item.getString("hero_a"),
                heroB = item.getString("hero_b"),
                sameTeamCount = item.getInt("same_team_count"),
                normalizedSynergyScore = item.getDouble("normalized_synergy_score"),
            )
        }
    }

    private fun loadResponseStats(): List<ResponseStatRow> {
        return loadStatArray(RESPONSE_STATS_PATH) { item ->
            ResponseStatRow(
                positionBucket = item.getString("position_bucket"),
                candidateHero = item.getString("candidate_hero"),
                enemyHero = item.getString("enemy_hero"),
                responseCount = item.getInt("response_count"),
                normalizedResponseScore = item.getDouble("normalized_response_score"),
            )
        }
    }

    private fun loadOpeningResponseStats(): List<OpeningResponseStatRow> {
        return loadStatArray(OPENING_RESPONSE_STATS_PATH) { item ->
            OpeningResponseStatRow(
                firstPickSide = item.getString("first_pick_side"),
                enemyOrder1Hero = item.getString("enemy_order_1_hero"),
                candidateHero = item.getString("candidate_hero"),
                responseCount = item.getInt("response_count"),
                normalizedResponseScore = item.getDouble("normalized_response_score"),
            )
        }
    }

    private fun <T> loadStatArray(path: String, mapper: (JSONObject) -> T): List<T> {
        val array = JSONArray(loadAssetText(path))
        return List(array.length()) { index -> mapper(array.getJSONObject(index)) }
    }

    private fun loadFinalBanConfig(): FinalBanConfig {
        val root = JSONObject(loadAssetText(FINAL_BAN_CONFIG_PATH))
        val responseWeights = root.getJSONObject("response_bucket_weights")
        val orderThreat = root.getJSONObject("order_threat_scores")
        val positionBucketWeights = root.optJSONObject("position_bucket_response_weights")?.let { bucketRoot ->
            bucketRoot.keys().asSequence().associateWith { bucketKey ->
                val nested = bucketRoot.getJSONObject(bucketKey)
                nested.keys().asSequence().associateWith { nested.getDouble(it) }
            }
        } ?: emptyMap()
        return FinalBanConfig(
            maxBannableEnemyPicks = root.getInt("max_bannable_enemy_picks"),
            minSynergyCount = root.getInt("min_synergy_count"),
            minResponseCount = root.getInt("min_response_count"),
            weightSynergy = root.getDouble("weight_synergy"),
            weightLackResponse = root.getDouble("weight_lack_response"),
            weightPositionThreat = root.getDouble("weight_position_threat"),
            banRateExponent = root.getDouble("ban_rate_exponent"),
            banRateRankWeights = root.getDoubleList("ban_rate_rank_weights"),
            responseBucketWeights = responseWeights.keys().asSequence().associateWith { key ->
                responseWeights.getDouble(key)
            },
            positionBucketResponseWeights = positionBucketWeights,
            orderThreatScores = orderThreat.keys().asSequence().associate { key ->
                key.toInt() to orderThreat.getDouble(key)
            },
            firstPickOrders = root.getIntList("first_pick_orders"),
            secondPickOrders = root.getIntList("second_pick_orders"),
            priorStrength = root.optDouble("prior_strength", 10.0),
            confidenceStrength = root.optDouble("confidence_strength", 20.0),
            maxHistoryWeight = root.optDouble("max_history_weight", 0.70),
            handledBy = root.getString("handled_by"),
        )
    }

    private fun loadFinalBanStats(): FinalBanStatsPayload {
        val root = JSONObject(loadAssetText(FINAL_BAN_STATS_PATH))
        val levels = mutableMapOf<String, Map<String, FinalBanCountEntry>>()
        val levelsRoot = root.getJSONObject("levels")
        levelsRoot.keys().forEach { levelName ->
            val levelObject = levelsRoot.getJSONObject(levelName)
            levels[levelName] = levelObject.keys().asSequence().associateWith { key ->
                val entry = levelObject.getJSONObject(key)
                FinalBanCountEntry(
                    banCount = entry.getInt("ban_count"),
                    eligibleCount = entry.getInt("eligible_count"),
                )
            }
        }
        val totals = root.getJSONObject("totals")
        return FinalBanStatsPayload(
            hasLabeledDecisions = root.optBoolean("has_labeled_decisions", false),
            decisionCount = root.optInt("decision_count", 0),
            levels = levels,
            totals = FinalBanCountEntry(
                banCount = totals.optInt("ban_count", 0),
                eligibleCount = totals.optInt("eligible_count", 0),
            ),
        )
    }

    private fun loadHeroAppearanceCounts(): Map<String, Int> {
        val root = JSONObject(loadAssetText(HERO_DETAILS_PATH))
        val heroes = root.getJSONArray("heroes")
        return buildMap {
            for (index in 0 until heroes.length()) {
                val hero = heroes.getJSONObject(index)
                put(hero.getString("code"), hero.optInt("appearance_count", 0))
            }
        }
    }

    private fun JSONObject.getStringList(key: String): List<String> {
        val array = optJSONArray(key) ?: JSONArray()
        return List(array.length()) { index -> array.getString(index) }
    }

    private fun JSONObject.getIntList(key: String): List<Int> {
        val array = getJSONArray(key)
        return List(array.length()) { index -> array.getInt(index) }
    }

    private fun JSONObject.getDoubleList(key: String): List<Double> {
        val array = getJSONArray(key)
        return List(array.length()) { index -> array.getDouble(index) }
    }

    companion object {
        private const val PREBAN_STATS_PATH = "data/preban_stats.json"
        private const val FIRST_PICK_RECORDS_PATH = "data/first_pick_records.json"
        private const val SYNERGY_STATS_PATH = "data/reranker_synergy_stats.json"
        private const val RESPONSE_STATS_PATH = "data/reranker_response_stats.json"
        private const val OPENING_RESPONSE_STATS_PATH = "data/reranker_opening_response_stats.json"
        private const val FINAL_BAN_CONFIG_PATH = "data/final_ban_config.json"
        private const val FINAL_BAN_STATS_PATH = "data/final_ban_stats.json"
        private const val HERO_DETAILS_PATH = "data/hero_details.json"
    }
}

data class PrebanStatsPayload(
    val byContext: Map<String, Map<String, Map<String, Int>>>,
    val combined: Map<String, Map<String, Int>>,
)

data class FirstPickRecord(
    val firstPickSide: String,
    val firstSidePreban: List<String>,
    val secondSidePreban: List<String>,
    val order1Hero: String,
    val season: String?,
)

data class SynergyStatRow(
    val heroA: String,
    val heroB: String,
    val sameTeamCount: Int,
    val normalizedSynergyScore: Double,
)

data class ResponseStatRow(
    val positionBucket: String,
    val candidateHero: String,
    val enemyHero: String,
    val responseCount: Int,
    val normalizedResponseScore: Double,
)

data class OpeningResponseStatRow(
    val firstPickSide: String,
    val enemyOrder1Hero: String,
    val candidateHero: String,
    val responseCount: Int,
    val normalizedResponseScore: Double,
)

data class FinalBanConfig(
    val maxBannableEnemyPicks: Int,
    val minSynergyCount: Int,
    val minResponseCount: Int,
    val weightSynergy: Double,
    val weightLackResponse: Double,
    val weightPositionThreat: Double,
    val banRateExponent: Double,
    val banRateRankWeights: List<Double>,
    val responseBucketWeights: Map<String, Double>,
    val positionBucketResponseWeights: Map<String, Map<String, Double>>,
    val orderThreatScores: Map<Int, Double>,
    val firstPickOrders: List<Int>,
    val secondPickOrders: List<Int>,
    val priorStrength: Double,
    val confidenceStrength: Double,
    val maxHistoryWeight: Double,
    val handledBy: String,
)

data class FinalBanCountEntry(
    val banCount: Int,
    val eligibleCount: Int,
)

data class FinalBanStatsPayload(
    val hasLabeledDecisions: Boolean,
    val decisionCount: Int,
    val levels: Map<String, Map<String, FinalBanCountEntry>>,
    val totals: FinalBanCountEntry,
)
