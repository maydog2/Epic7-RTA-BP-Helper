package com.e7.bphelper

class InferenceArrayBuilder(
    private val metadata: EncoderMetadata,
) {
    fun build(input: DraftRecommendInput): InferenceTensors {
        val normalizedRule = normalizeWarfareRules(input.warfareRules)
        val warfareRuleId = if (normalizedRule == WARFARE_RULE_ANY) {
            metadata.warfareRuleId(metadata.warfareRuleOrder.first())
        } else {
            metadata.warfareRuleId(normalizedRule)
        }

        val userPicks = normalizePicks(input.userPicks)
        val enemyPicks = normalizePicks(input.enemyPicks)
        val allyPreban = normalizePicks(input.allyPreban).take(metadata.maxPreban)
        val enemyPreban = normalizePicks(input.enemyPreban).take(metadata.maxPreban)

        val (ordered, firstPickSide) = interleaveApiPicks(
            userPicks = userPicks,
            enemyPicks = enemyPicks,
            firstPickTeam = input.firstPickTeam,
        )

        val sideOrder = if (firstPickSide == "ally") {
            metadata.sideOrderAllyFirst
        } else {
            metadata.sideOrderEnemyFirst
        }
        val firstPickSideToken = if (firstPickSide == "ally") {
            metadata.allySideId
        } else {
            metadata.enemySideId
        }
        val isFirstPickSlot = IntArray(metadata.draftSlots) { index ->
            if (sideOrder[index] == firstPickSideToken) 1 else 0
        }
        val draftOrderIds = IntArray(metadata.draftSlots) { index -> index + 1 }

        val heroIds = IntArray(metadata.draftSlots)
        val sideIds = IntArray(metadata.draftSlots)
        ordered.forEachIndexed { slotIndex, (heroCode, sideName) ->
            heroIds[slotIndex] = metadata.heroId(if (heroCode == "unknown") null else heroCode)
            sideIds[slotIndex] = if (sideName == "ally") metadata.allySideId else metadata.enemySideId
        }

        val roleIds = IntArray(metadata.draftSlots) { index ->
            metadata.roleIdForHeroId(heroIds[index])
        }
        val elementIds = IntArray(metadata.draftSlots) { index ->
            metadata.elementIdForHeroId(heroIds[index])
        }

        val allyPrebanIds = padHeroIds(allyPreban.map(metadata::heroId))
        val enemyPrebanIds = padHeroIds(enemyPreban.map(metadata::heroId))
        val allyPrebanRoleIds = IntArray(metadata.maxPreban) { index ->
            metadata.roleIdForHeroId(allyPrebanIds[index])
        }
        val allyPrebanElementIds = IntArray(metadata.maxPreban) { index ->
            metadata.elementIdForHeroId(allyPrebanIds[index])
        }
        val enemyPrebanRoleIds = IntArray(metadata.maxPreban) { index ->
            metadata.roleIdForHeroId(enemyPrebanIds[index])
        }
        val enemyPrebanElementIds = IntArray(metadata.maxPreban) { index ->
            metadata.elementIdForHeroId(enemyPrebanIds[index])
        }

        val candidateMask = buildCandidateMask(
            draftHeroIds = heroIds,
            allyPrebanIds = allyPrebanIds,
            enemyPrebanIds = enemyPrebanIds,
        )

        val draftStep = ordered.size
        val nextOrder = minOf(draftStep + 1, metadata.draftSlots)
        val currentSideId = if (draftStep < metadata.draftSlots) {
            sideOrder[draftStep]
        } else {
            sideOrder.last()
        }
        val currentPositionBucketId = metadata.draftPositionBucketIds[nextOrder - 1]

        return InferenceTensors(
            draftHeroIds = heroIds,
            draftRoleIds = roleIds,
            draftElementIds = elementIds,
            draftSideIds = sideIds,
            draftOrderIds = draftOrderIds,
            draftPositionBucketIds = metadata.draftPositionBucketIds.copyOf(),
            isFirstPickSlot = isFirstPickSlot,
            allyPrebanIds = allyPrebanIds,
            allyPrebanRoleIds = allyPrebanRoleIds,
            allyPrebanElementIds = allyPrebanElementIds,
            enemyPrebanIds = enemyPrebanIds,
            enemyPrebanRoleIds = enemyPrebanRoleIds,
            enemyPrebanElementIds = enemyPrebanElementIds,
            candidateMask = candidateMask,
            currentPositionBucketId = currentPositionBucketId,
            currentOrder = nextOrder,
            currentSideId = currentSideId,
            warfareRuleId = warfareRuleId,
        )
    }

    private fun normalizePicks(picks: List<String>): List<String> {
        val playable = metadata.heroToId.keys
            .filter { it != "<PAD>" && it != "<UNK>" }
            .toSet()
        return picks.map { pick -> if (pick in playable) pick else "unknown" }
    }

    private fun padHeroIds(ids: List<Int>): IntArray {
        val padded = IntArray(metadata.maxPreban) { metadata.padHeroId }
        ids.take(metadata.maxPreban).forEachIndexed { index, heroId ->
            padded[index] = heroId
        }
        return padded
    }

    private fun buildCandidateMask(
        draftHeroIds: IntArray,
        allyPrebanIds: IntArray,
        enemyPrebanIds: IntArray,
    ): FloatArray {
        val mask = FloatArray(metadata.numHeroes) { 1f }
        val unavailable = mutableSetOf<Int>()
        draftHeroIds.forEach { heroId -> unavailable.add(heroId) }
        allyPrebanIds.forEach { heroId -> unavailable.add(heroId) }
        enemyPrebanIds.forEach { heroId -> unavailable.add(heroId) }
        unavailable.remove(metadata.padHeroId)
        unavailable.forEach { heroId ->
            if (heroId in 0 until metadata.numHeroes) {
                mask[heroId] = 0f
            }
        }
        mask[metadata.padHeroId] = 0f
        mask[metadata.unkHeroId] = 1f
        return mask
    }

    private fun interleaveApiPicks(
        userPicks: List<String>,
        enemyPicks: List<String>,
        firstPickTeam: String,
    ): Pair<List<Pair<String, String>>, String> {
        val trimmed = if (firstPickTeam == "My Team") {
            trimDraftPicks(userPicks, enemyPicks)
        } else {
            val reversed = trimDraftPicks(enemyPicks, userPicks)
            reversed.second to reversed.first
        }
        val trimmedUser = trimmed.first
        val trimmedEnemy = trimmed.second
        val firstPickSide = if (firstPickTeam == "My Team") "ally" else "enemy"

        val ordered = mutableListOf<Pair<String, String>>()
        var userIdx = 0
        var enemyIdx = 0
        repeat(trimmedUser.size + trimmedEnemy.size) { slot ->
            if (firstPickSide == "ally") {
                if (slot in metadata.firstPickSlotIndices) {
                    ordered += trimmedUser[userIdx] to "ally"
                    userIdx += 1
                } else {
                    ordered += trimmedEnemy[enemyIdx] to "enemy"
                    enemyIdx += 1
                }
            } else {
                if (slot in metadata.firstPickSlotIndices) {
                    ordered += trimmedEnemy[enemyIdx] to "enemy"
                    enemyIdx += 1
                } else {
                    ordered += trimmedUser[userIdx] to "ally"
                    userIdx += 1
                }
            }
        }
        return ordered to firstPickSide
    }

    private fun trimDraftPicks(
        firstTeamPicks: List<String>,
        nonFirstTeamPicks: List<String>,
    ): Pair<List<String>, List<String>> {
        val maxNonFirstByFirstCount = intArrayOf(0, 2, 2, 4, 4, 5)
        val maxFirstByNonFirstCount = intArrayOf(1, 1, 3, 3, 5, 5, 5)

        var first = firstTeamPicks.take(5)
        var nonFirst = nonFirstTeamPicks.take(5)

        if (first.size < 6) {
            val limit = maxNonFirstByFirstCount.getOrElse(first.size) { 5 }
            nonFirst = nonFirst.take(limit)
        }
        if (nonFirst.size < 6) {
            val limit = maxFirstByNonFirstCount.getOrElse(nonFirst.size) { 5 }
            first = first.take(limit)
        }
        return first.take(5) to nonFirst.take(5)
    }

    private fun normalizeWarfareRules(rawValue: String?): String {
        val text = rawValue?.trim().orEmpty()
        if (text.isEmpty() || text.equals(WARFARE_RULE_ANY, ignoreCase = true)) {
            return WARFARE_RULE_ANY
        }
        require(text in metadata.warfareRuleOrder) {
            "Invalid warfare_rules value '$rawValue'"
        }
        return text
    }

    companion object {
        const val WARFARE_RULE_ANY = "ANY"
    }
}
