package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class FinalBanRecommender(
    private val artifacts: RecommendationArtifacts,
    private val validHeroes: Set<String>,
) {
    private val config: FinalBanConfig = artifacts.finalBanConfig
    private val stats: FinalBanStatsPayload = artifacts.finalBanStats
    private val synergyLookup: Map<Pair<String, String>, SynergyEntry> by lazy { buildSynergyLookup() }
    private val responseLookup: Map<Triple<String, String, String>, ResponseEntry> by lazy { buildResponseLookup() }
    private val responseEvidence: Set<String> by lazy {
        responseLookup.values.map { it.enemyHero }.toSet()
    }

    fun recommend(
        userTeamPicks: List<String>,
        enemyTeamPicks: List<String>,
        firstPickTeam: String,
        allyPreban: List<String> = emptyList(),
        enemyPreban: List<String> = emptyList(),
        warfareRules: String = "ANY",
        topK: Int = config.maxBannableEnemyPicks,
    ): JSONObject {
        val allyPicks = DraftPositionUtils.deriveOrderedPicks(userTeamPicks, firstPickTeam, "ally")
        val enemyPicks = DraftPositionUtils.deriveOrderedPicks(enemyTeamPicks, firstPickTeam, "enemy")
        val (candidates, filteredOut) = buildBanCandidates(
            allyPicks = allyPicks,
            enemyPicks = enemyPicks,
            firstPickTeam = firstPickTeam,
            allyPreban = allyPreban,
            enemyPreban = enemyPreban,
        )

        val enemyContext = enemyPicks.filter { pick -> isValidHero(pick.hero) }
        val actorPickOrder = if (firstPickTeam == "My Team") "first" else "second"
        val recommendations = candidates.map { pick ->
            val hero = pick.hero
            val order = pick.order
            val synergyScore = enemySynergyCoreScore(hero, enemyContext)
            val (lackScore, teamCoverage) = allyLackResponseScore(hero, allyPicks, pick.positionBucket)
            val threatScore = enemyPositionThreatScore(order)
            val formulaScore = min(
                max(
                    config.weightSynergy * synergyScore +
                        config.weightLackResponse * lackScore +
                        config.weightPositionThreat * threatScore,
                    0.0,
                ),
                1.0,
            )
            val historicalLookup = lookupHistoricalBanRate(
                actorPickOrder = actorPickOrder,
                warfareRules = warfareRules,
                positionBucket = pick.positionBucket,
                hero = hero,
            )
            val historyWeight = historicalLookup.historyWeight
            val banScore = min(
                max(
                    historyWeight * historicalLookup.historicalScore + (1.0 - historyWeight) * formulaScore,
                    0.0,
                ),
                1.0,
            )

            BanRecommendation(
                hero = hero,
                order = order,
                positionBucket = pick.positionBucket,
                banScore = banScore,
                formulaScore = formulaScore,
                historicalLookup = historicalLookup,
                enemySynergyCoreScore = synergyScore,
                allyLackResponseScore = lackScore,
                enemyPositionThreatScore = threatScore,
                teamCoverage = teamCoverage,
            )
        }.sortedWith(
            compareByDescending<BanRecommendation> { it.banScore }
                .thenByDescending { it.enemySynergyCoreScore }
                .thenByDescending { it.allyLackResponseScore }
                .thenByDescending { it.enemyPositionThreatScore }
                .thenBy { it.order }
                .thenBy { it.hero },
        )

        val limited = recommendations.take(max(topK, 0))
        val topRates = scoreToDisplayRates(limited)
        val topHeroes = JSONArray()
        val topRatesArray = JSONArray()
        val recommendationArray = JSONArray()

        limited.forEachIndexed { index, item ->
            topHeroes.put(item.hero)
            topRatesArray.put(topRates[index])
            recommendationArray.put(
                JSONObject()
                    .put("hero", item.hero)
                    .put("order", item.order)
                    .put("position_bucket", item.positionBucket)
                    .put("ban_score", round6(item.banScore))
                    .put("formula_score", round6(item.formulaScore))
                    .put("historical_ban_rate", round6(item.historicalLookup.historicalScore))
                    .put("historical_ban_count", item.historicalLookup.banCount)
                    .put("historical_eligible_count", item.historicalLookup.eligibleCount)
                    .put("historical_context_level", item.historicalLookup.contextLevel)
                    .put("historical_confidence", round6(item.historicalLookup.confidence))
                    .put("historical_weight", round6(item.historicalLookup.historyWeight))
                    .put("enemy_synergy_core_score", round6(item.enemySynergyCoreScore))
                    .put("ally_lack_response_score", round6(item.allyLackResponseScore))
                    .put("enemy_position_threat_score", round6(item.enemyPositionThreatScore))
                    .put("reasons", JSONArray(buildReasons(item)))
                    .put(
                        "debug",
                        JSONObject()
                            .put("team_response_coverage", round6(item.teamCoverage))
                            .put("historical_parent_rate", round6(item.historicalLookup.parentRate))
                            .put("filtered_out", JSONArray(filteredOut.map { it.toJson() })),
                    ),
            )
        }

        return JSONObject()
            .put("phase", "ban")
            .put("handled_by", config.handledBy)
            .put("top_10_heroes", topHeroes)
            .put("top_10_rates", topRatesArray)
            .put("recommendations", recommendationArray)
    }

    private fun lookupHistoricalBanRate(
        actorPickOrder: String,
        warfareRules: String,
        positionBucket: String,
        hero: String,
    ): HistoricalLookupResult {
        if (!stats.hasLabeledDecisions) {
            return HistoricalLookupResult(
                historicalScore = 0.0,
                banCount = 0,
                eligibleCount = 0,
                contextLevel = null,
                confidence = 0.0,
                historyWeight = 0.0,
                parentRate = globalBaselineRate(),
            )
        }

        val normalizedRule = warfareRules.trim().takeIf { it.isNotEmpty() && !it.equals("ANY", ignoreCase = true) }
        val chain = buildList {
            if (normalizedRule != null) {
                add("level_1" to encodeLevelKey(actorPickOrder, normalizedRule, positionBucket, hero))
            }
            add("level_2" to encodeLevelKey(actorPickOrder, positionBucket, hero))
            add("level_3" to encodeLevelKey(actorPickOrder, hero))
            add("level_4" to hero)
        }

        for ((levelName, key) in chain) {
            val entry = stats.levels[levelName]?.get(key) ?: continue
            if (entry.eligibleCount <= 0) {
                continue
            }
            val parentRate = parentRateForLevel(
                levelName = levelName,
                actorPickOrder = actorPickOrder,
                hero = hero,
            )
            val historicalScore = smoothedRate(
                banCount = entry.banCount,
                eligibleCount = entry.eligibleCount,
                parentRate = parentRate,
            )
            val confidence = entry.eligibleCount / (entry.eligibleCount + config.confidenceStrength)
            val historyWeight = config.maxHistoryWeight * confidence
            return HistoricalLookupResult(
                historicalScore = min(max(historicalScore, 0.0), 1.0),
                banCount = entry.banCount,
                eligibleCount = entry.eligibleCount,
                contextLevel = CONTEXT_LEVEL_LABELS[levelName],
                confidence = confidence,
                historyWeight = historyWeight,
                parentRate = parentRate,
            )
        }

        return HistoricalLookupResult(
            historicalScore = 0.0,
            banCount = 0,
            eligibleCount = 0,
            contextLevel = null,
            confidence = 0.0,
            historyWeight = 0.0,
            parentRate = globalBaselineRate(),
        )
    }

    private fun parentRateForLevel(levelName: String, actorPickOrder: String, hero: String): Double {
        val baseline = globalBaselineRate()
        return when (levelName) {
            "level_1", "level_2" -> {
                val level3 = stats.levels["level_3"]?.get(encodeLevelKey(actorPickOrder, hero))
                if (level3 != null && level3.eligibleCount > 0) {
                    val level4 = stats.levels["level_4"]?.get(hero)
                    val parent = if (level4 != null && level4.eligibleCount > 0) {
                        smoothedRate(level4.banCount, level4.eligibleCount, baseline)
                    } else {
                        baseline
                    }
                    smoothedRate(level3.banCount, level3.eligibleCount, parent)
                } else {
                    val level4 = stats.levels["level_4"]?.get(hero)
                    if (level4 != null && level4.eligibleCount > 0) {
                        smoothedRate(level4.banCount, level4.eligibleCount, baseline)
                    } else {
                        baseline
                    }
                }
            }
            "level_3" -> {
                val level4 = stats.levels["level_4"]?.get(hero)
                if (level4 != null && level4.eligibleCount > 0) {
                    smoothedRate(level4.banCount, level4.eligibleCount, baseline)
                } else {
                    baseline
                }
            }
            else -> baseline
        }
    }

    private fun globalBaselineRate(): Double {
        val eligible = stats.totals.eligibleCount
        if (eligible <= 0) {
            return 0.25
        }
        return stats.totals.banCount.toDouble() / eligible.toDouble()
    }

    private fun smoothedRate(banCount: Int, eligibleCount: Int, parentRate: Double): Double {
        return (banCount + config.priorStrength * parentRate) / (eligibleCount + config.priorStrength)
    }

    private fun encodeLevelKey(vararg parts: String): String = parts.joinToString("|")

    private fun buildBanCandidates(
        allyPicks: List<OrderedPick>,
        enemyPicks: List<OrderedPick>,
        firstPickTeam: String,
        allyPreban: List<String>,
        enemyPreban: List<String>,
    ): Pair<List<OrderedPick>, List<FilteredOutPick>> {
        val protectedOrder = DraftPositionUtils.enemyProtectedOrder(firstPickTeam)
        val allyHeroSet = allyPicks.map { it.hero }.toSet()
        val prebanSet = allyPreban.toSet() + enemyPreban.toSet()
        val candidates = mutableListOf<OrderedPick>()
        val filteredOut = mutableListOf<FilteredOutPick>()

        enemyPicks.forEach { pick ->
            val reason = when {
                pick.order == protectedOrder -> "protected_order"
                !isValidHero(pick.hero) -> "invalid_hero"
                pick.hero in allyHeroSet -> "ally_pick_overlap"
                pick.hero in prebanSet -> "prebanned"
                else -> null
            }
            if (reason != null) {
                filteredOut += FilteredOutPick(pick.hero, pick.order, reason)
            } else {
                candidates += pick
            }
        }
        return candidates to filteredOut
    }

    private fun enemySynergyCoreScore(
        candidateHero: String,
        enemyContext: List<OrderedPick>,
    ): Double {
        val matchedScores = enemyContext.mapNotNull { contextPick ->
            if (contextPick.hero.isBlank() || contextPick.hero == candidateHero) {
                return@mapNotNull null
            }
            synergyLookup[candidateHero to contextPick.hero]?.score
        }
        return combineSynergyScores(matchedScores)
    }

    private fun responseBucketWeights(positionBucket: String): Map<String, Double> {
        return config.positionBucketResponseWeights[positionBucket] ?: config.responseBucketWeights
    }

    private fun allyLackResponseScore(
        enemyCandidate: String,
        allyPicks: List<OrderedPick>,
        positionBucket: String,
    ): Pair<Double, Double> {
        val bucketWeights = responseBucketWeights(positionBucket)
        val allyCoverages = mutableListOf<Double>()
        allyPicks.forEach { pick ->
            if (!isValidHero(pick.hero)) {
                return@forEach
            }
            val coverage = allyResponseCoverage(pick.hero, enemyCandidate, bucketWeights)
            if (coverage > 0.0) {
                allyCoverages += coverage
            }
        }

        if (enemyCandidate !in responseEvidence) {
            return 0.50 to 0.0
        }
        if (allyCoverages.isEmpty()) {
            return 1.0 to 0.0
        }

        val sorted = allyCoverages.sortedDescending()
        val teamCoverage = if (sorted.size == 1) {
            sorted[0]
        } else {
            min(1.0, 0.75 * sorted[0] + 0.25 * sorted[1])
        }
        val lackScore = min(max(1.0 - teamCoverage, 0.0), 1.0)
        return lackScore to teamCoverage
    }

    private fun allyResponseCoverage(
        allyHero: String,
        enemyCandidate: String,
        bucketWeights: Map<String, Double>,
    ): Double {
        var weightedSum = 0.0
        var weightTotal = 0.0
        bucketWeights.forEach { (bucket, bucketWeight) ->
            val entry = responseLookup[Triple(bucket, allyHero, enemyCandidate)] ?: return@forEach
            weightedSum += bucketWeight * entry.score
            weightTotal += bucketWeight
        }
        if (weightTotal <= 0.0) {
            return 0.0
        }
        return min(weightedSum / weightTotal, 1.0)
    }

    private fun enemyPositionThreatScore(order: Int): Double {
        return config.orderThreatScores[order] ?: 0.50
    }

    private fun combineSynergyScores(scores: List<Double>): Double {
        if (scores.isEmpty()) {
            return 0.0
        }
        val average = scores.average()
        val maximum = scores.maxOrNull() ?: 0.0
        return min(max(0.60 * average + 0.40 * maximum, 0.0), 1.0)
    }

    private fun buildReasons(item: BanRecommendation): List<String> {
        val reasons = mutableListOf<String>()
        val lookup = item.historicalLookup
        if (lookup.contextLevel != null && lookup.confidence >= 0.35 && lookup.historicalScore >= 0.30) {
            reasons += "Historically banned often in similar draft contexts " +
                "(${lookup.banCount}/${lookup.eligibleCount} eligible picks)"
        } else if (lookup.contextLevel != null && lookup.confidence > 0.0) {
            reasons += "Limited historical ban evidence; blending with lineup formula"
        } else if (lookup.historyWeight <= 0.0) {
            reasons += "No labeled final-ban history; using lineup formula only"
        }
        if (item.enemySynergyCoreScore >= 0.45) {
            reasons += "Strong synergy with enemy team core"
        }
        if (item.allyLackResponseScore >= 0.65) {
            reasons += "Ally team lacks known position-bucket responses"
        } else if (item.allyLackResponseScore <= 0.35) {
            reasons += "Ally team already has response coverage"
        }
        if (item.enemyPositionThreatScore >= 0.85) {
            reasons += "High late-pick threat at order ${item.order}"
        } else if (item.enemyPositionThreatScore >= 0.70) {
            reasons += "Mid/late draft threat at order ${item.order}"
        }
        if (reasons.isEmpty()) {
            reasons += "Balanced ban value from draft-position and team-context stats"
        }
        return reasons
    }

    private fun scoreToDisplayRates(recommendations: List<BanRecommendation>): List<Double> {
        val sharpenedScores = recommendations.mapIndexed { index, item ->
            max(item.banScore, 0.0).pow(config.banRateExponent) *
                config.banRateRankWeights[min(index, config.banRateRankWeights.lastIndex)]
        }
        val totalScore = sharpenedScores.sum()
        if (totalScore <= 0.0) {
            return List(recommendations.size) { 0.0 }
        }
        return sharpenedScores.map { score -> score / totalScore * 100.0 }
    }

    private fun buildSynergyLookup(): Map<Pair<String, String>, SynergyEntry> {
        val lookup = mutableMapOf<Pair<String, String>, SynergyEntry>()
        artifacts.synergyStats.forEach { row ->
            if (row.sameTeamCount < config.minSynergyCount) {
                return@forEach
            }
            val entry = SynergyEntry(
                score = min(max(row.normalizedSynergyScore, 0.0), 1.0),
                sameTeamCount = row.sameTeamCount,
            )
            lookup[row.heroA to row.heroB] = entry
            lookup[row.heroB to row.heroA] = entry
        }
        return lookup
    }

    private fun buildResponseLookup(): Map<Triple<String, String, String>, ResponseEntry> {
        val lookup = mutableMapOf<Triple<String, String, String>, ResponseEntry>()
        val supportedBuckets = config.responseBucketWeights.keys +
            config.positionBucketResponseWeights.values.flatMap { it.keys }
        artifacts.responseStats.forEach { row ->
            if (row.responseCount < config.minResponseCount) {
                return@forEach
            }
            if (row.positionBucket !in supportedBuckets) {
                return@forEach
            }
            lookup[Triple(row.positionBucket, row.candidateHero, row.enemyHero)] = ResponseEntry(
                score = min(max(row.normalizedResponseScore, 0.0), 1.0),
                responseCount = row.responseCount,
                enemyHero = row.enemyHero,
            )
        }
        return lookup
    }

    private fun isValidHero(hero: String?): Boolean {
        if (hero.isNullOrBlank() || hero in INVALID_HERO_TOKENS) {
            return false
        }
        return hero in validHeroes
    }

    private fun round6(value: Double): Double {
        return "%.6f".format(value).toDouble()
    }

    private data class SynergyEntry(
        val score: Double,
        val sameTeamCount: Int,
    )

    private data class ResponseEntry(
        val score: Double,
        val responseCount: Int,
        val enemyHero: String,
    )

    private data class HistoricalLookupResult(
        val historicalScore: Double,
        val banCount: Int,
        val eligibleCount: Int,
        val contextLevel: String?,
        val confidence: Double,
        val historyWeight: Double,
        val parentRate: Double,
    )

    private data class BanRecommendation(
        val hero: String,
        val order: Int,
        val positionBucket: String,
        val banScore: Double,
        val formulaScore: Double,
        val historicalLookup: HistoricalLookupResult,
        val enemySynergyCoreScore: Double,
        val allyLackResponseScore: Double,
        val enemyPositionThreatScore: Double,
        val teamCoverage: Double,
    )

    private data class FilteredOutPick(
        val hero: String,
        val order: Int,
        val reason: String,
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("hero", hero)
                .put("order", order)
                .put("reason", reason)
        }
    }

    companion object {
        private val INVALID_HERO_TOKENS = setOf("", "unknown", "<PAD>", "<UNK>")
        private val CONTEXT_LEVEL_LABELS = mapOf(
            "level_1" to "actor_rule_bucket_hero",
            "level_2" to "actor_bucket_hero",
            "level_3" to "actor_hero",
            "level_4" to "hero_global",
        )
    }
}
