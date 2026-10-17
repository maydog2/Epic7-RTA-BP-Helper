package com.e7.bphelper

data class DraftRecommendInput(
    val userPicks: List<String> = emptyList(),
    val enemyPicks: List<String> = emptyList(),
    val allyPreban: List<String> = emptyList(),
    val enemyPreban: List<String> = emptyList(),
    val firstPickTeam: String = "My Team",
    val warfareRules: String = "ANY",
    val topK: Int = 10,
)

data class RankedHero(
    val heroCode: String,
    val rate: Double,
)

data class RecommendationResult(
    val topHeroes: List<RankedHero>,
)

data class TflitePickResult(
    val topHeroes: List<RankedHero>,
    val modelScores: Map<String, Double>,
) {
    fun toRecommendationResult(): RecommendationResult {
        return RecommendationResult(topHeroes = topHeroes)
    }
}
