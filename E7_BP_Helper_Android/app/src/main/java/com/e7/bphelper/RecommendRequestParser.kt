package com.e7.bphelper

import android.net.Uri

class RecommendRequestParser(
    private val metadata: EncoderMetadata,
) {
    fun parse(uri: Uri): DraftRecommendInput {
        val userPicks = parsePicks(uri.getQueryParameter("user_picks"))
        val enemyPicks = parsePicks(uri.getQueryParameter("enemy_picks"))
        val allyPreban = parsePicks(uri.getQueryParameter("ally_preban"))
        val enemyPreban = parsePicks(uri.getQueryParameter("enemy_preban"))
        val firstPickTeam = uri.getQueryParameter("first_pick_team")?.trim().orEmpty().ifEmpty { "My Team" }
        val warfareRules = uri.getQueryParameter("warfare_rules")?.trim().orEmpty().ifEmpty { "ANY" }

        if (firstPickTeam != "My Team" && firstPickTeam != "Enemy Team") {
            throw RecommendRequestException("first_pick_team must be 'My Team' or 'Enemy Team'")
        }

        validateWarfareRules(warfareRules)
        // Hero codes outside the Transformer vocab are normalized to "unknown" in
        // InferenceArrayBuilder.normalizePicks(), matching backend normalize_picks().

        return DraftRecommendInput(
            userPicks = userPicks,
            enemyPicks = enemyPicks,
            allyPreban = allyPreban,
            enemyPreban = enemyPreban,
            firstPickTeam = firstPickTeam,
            warfareRules = warfareRules,
            topK = metadata.defaultTopK,
        )
    }

    private fun parsePicks(rawValue: String?): List<String> {
        if (rawValue.isNullOrBlank()) {
            return emptyList()
        }
        return rawValue.split(",").map { it.trim() }.filter { it.isNotEmpty() }
    }

    private fun validateWarfareRules(rawValue: String) {
        if (rawValue.equals("ANY", ignoreCase = true)) {
            return
        }
        if (rawValue !in metadata.warfareRuleOrder) {
            throw RecommendRequestException(
                "Invalid warfare_rules value '$rawValue'; expected ANY or one of ${metadata.warfareRuleOrder}",
            )
        }
    }
}

class RecommendRequestException(message: String) : Exception(message)
