package com.e7.bphelper

object DraftPositionUtils {
    const val FIRST_PICK_SIDE_ALLY = "ally"
    const val FIRST_PICK_SIDE_ENEMY = "enemy"

    private val firstPickOrders = intArrayOf(1, 4, 5, 8, 9)
    private val secondPickOrders = intArrayOf(2, 3, 6, 7, 10)

    fun parseFirstPickSide(firstPickTeam: String?): String? {
        if (firstPickTeam.isNullOrBlank()) {
            return null
        }
        return when (firstPickTeam.trim().lowercase()) {
            "my team", "ally", "user" -> FIRST_PICK_SIDE_ALLY
            "enemy team", "enemy" -> FIRST_PICK_SIDE_ENEMY
            else -> null
        }
    }

    fun getPositionBucket(order: Int): String {
        return when (order) {
            1 -> "1"
            2, 3 -> "2_3"
            4 -> "4"
            5 -> "5_protected"
            6 -> "6_protected"
            7 -> "7"
            8, 9 -> "8_9"
            10 -> "10"
            else -> throw IllegalArgumentException("Invalid draft order: $order")
        }
    }

    fun deriveOrderedPicks(
        heroCodes: List<String>,
        firstPickTeam: String,
        side: String,
    ): List<OrderedPick> {
        val isFirstPick = if (side == FIRST_PICK_SIDE_ALLY) {
            firstPickTeam == "My Team"
        } else {
            firstPickTeam == "Enemy Team"
        }
        val orders = if (isFirstPick) firstPickOrders else secondPickOrders
        return heroCodes.take(5).mapIndexed { index, hero ->
            val order = orders[index]
            OrderedPick(
                hero = hero,
                order = order,
                positionBucket = getPositionBucket(order),
            )
        }
    }

    fun enemyProtectedOrder(firstPickTeam: String): Int {
        return if (firstPickTeam == "Enemy Team") 5 else 6
    }

    fun unavailableHeroesFromDraft(input: DraftRecommendInput): Set<String> {
        return buildSet {
            input.userPicks.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.enemyPicks.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.allyPreban.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
            input.enemyPreban.forEach { code -> if (code.isNotBlank() && code != "unknown") add(code) }
        }
    }

    fun isFirstPickRecommendationTurn(
        firstPickTeam: String,
        userTeamPicks: List<String>,
        enemyTeamPicks: List<String>,
    ): Boolean {
        return when (firstPickTeam) {
            "My Team" -> userTeamPicks.isEmpty()
            "Enemy Team" -> enemyTeamPicks.isEmpty()
            else -> false
        }
    }
}

data class OrderedPick(
    val hero: String,
    val order: Int,
    val positionBucket: String,
)
