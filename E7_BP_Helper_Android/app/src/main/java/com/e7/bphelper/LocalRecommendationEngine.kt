package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject

class LocalRecommendationEngine(
    private val metadata: EncoderMetadata,
    private val tfliteEngine: TfliteRecommendEngine,
    private val artifacts: RecommendationArtifacts,
) {
    private val arrayBuilder = InferenceArrayBuilder(metadata)
    private val ranker = CandidateRanker(metadata)
    private val firstPickRecommender = FirstPickRecommender(artifacts)
    private val finalBanRecommender = FinalBanRecommender(
        artifacts = artifacts,
        validHeroes = artifacts.heroAppearanceCounts.keys,
    )
    private val reranker = RecommendationReranker(artifacts)

    fun recommend(input: DraftRecommendInput): JSONObject {
        if (input.userPicks.size >= 5 && input.enemyPicks.size >= 5) {
            return finalBanRecommender.recommend(
                userTeamPicks = input.userPicks,
                enemyTeamPicks = input.enemyPicks,
                firstPickTeam = input.firstPickTeam,
                allyPreban = input.allyPreban,
                enemyPreban = input.enemyPreban,
                warfareRules = input.warfareRules,
                topK = 4,
            )
        }

        val unavailableHeroes = DraftPositionUtils.unavailableHeroesFromDraft(input)
        if (DraftPositionUtils.isFirstPickRecommendationTurn(
                input.firstPickTeam,
                input.userPicks,
                input.enemyPicks,
            )
        ) {
            val firstPickSide = DraftPositionUtils.parseFirstPickSide(input.firstPickTeam)
                ?: return firstPickRecommender.recommend(
                    allyPreban = input.allyPreban,
                    enemyPreban = input.enemyPreban,
                    firstPickSide = "invalid",
                    excludedHeroes = unavailableHeroes,
                    topK = input.topK,
                )
            return firstPickRecommender.recommend(
                allyPreban = input.allyPreban,
                enemyPreban = input.enemyPreban,
                firstPickSide = firstPickSide,
                excludedHeroes = unavailableHeroes,
                topK = input.topK,
            )
        }

        val tensors = arrayBuilder.build(input)
        val rawResult = tfliteEngine.predictWithScores(input, tensors)
        val positionBucket = metadata.positionBucketForId(tensors.currentPositionBucketId)

        if (positionBucket == "1" || rawResult.topHeroes.isEmpty()) {
            return buildPickPayload(
                topHeroes = rawResult.topHeroes,
                rerankerApplied = false,
            )
        }

        val modelScores = rawResult.topHeroes.associate { hero ->
            hero.heroCode to (rawResult.modelScores[hero.heroCode] ?: 0.0)
        }
        val rerankResult = reranker.rerankCandidates(
            candidates = rawResult.topHeroes.map { it.heroCode },
            modelScores = modelScores,
            allyPicks = input.userPicks,
            enemyPicks = input.enemyPicks,
            unavailableHeroes = unavailableHeroes,
            positionBucket = positionBucket,
            firstPickTeam = input.firstPickTeam,
            topK = input.topK,
        )

        val topHeroes = rerankResult.topHeroes.mapIndexed { index, heroCode ->
            RankedHero(
                heroCode = heroCode,
                rate = rerankResult.topRates.getOrElse(index) { 0.0 },
            )
        }
        return buildPickPayload(
            topHeroes = topHeroes,
            rerankerApplied = rerankResult.usedReranker,
        )
    }

    private fun buildPickPayload(
        topHeroes: List<RankedHero>,
        rerankerApplied: Boolean,
    ): JSONObject {
        val topHeroesArray = JSONArray()
        val topRatesArray = JSONArray()
        topHeroes.forEach { hero ->
            topHeroesArray.put(hero.heroCode)
            topRatesArray.put(hero.rate)
        }
        return JSONObject()
            .put("phase", "pick")
            .put("top_10_heroes", topHeroesArray)
            .put("top_10_rates", topRatesArray)
            .put("reranker_enabled", true)
            .put("reranker_applied", rerankerApplied)
    }
}
