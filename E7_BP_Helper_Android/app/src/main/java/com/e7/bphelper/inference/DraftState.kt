package com.e7.bphelper.inference

data class DraftState(
    val userPicks: List<String>,
    val enemyPicks: List<String>,
    val allyPreban: List<String> = emptyList(),
    val enemyPreban: List<String> = emptyList(),
    val firstPickTeam: String,
    val warfareRules: String = "ANY",
)

data class RecommendationItem(
    val heroCode: String,
    val rate: Double,
)

data class RecommendationResult(
    val topHeroes: List<String>,
    val topRates: List<Double>,
    val items: List<RecommendationItem>,
)

data class InferenceArrays(
    val draftHeroIds: IntArray,
    val draftRoleIds: IntArray,
    val draftElementIds: IntArray,
    val draftSideIds: IntArray,
    val draftOrderIds: IntArray,
    val draftPositionBucketIds: IntArray,
    val isFirstPickSlot: IntArray,
    val allyPrebanIds: IntArray,
    val allyPrebanRoleIds: IntArray,
    val allyPrebanElementIds: IntArray,
    val enemyPrebanIds: IntArray,
    val enemyPrebanRoleIds: IntArray,
    val enemyPrebanElementIds: IntArray,
    val draftFilledMask: FloatArray,
    val candidateMask: FloatArray,
    val currentPositionBucketId: IntArray,
    val currentOrder: IntArray,
    val currentSideId: IntArray,
    val warfareRuleId: IntArray,
) {
    fun toInputMap(): Map<String, Any> = mapOf(
        "draft_hero_ids" to draftHeroIds,
        "draft_role_ids" to draftRoleIds,
        "draft_element_ids" to draftElementIds,
        "draft_side_ids" to draftSideIds,
        "draft_order_ids" to draftOrderIds,
        "draft_position_bucket_ids" to draftPositionBucketIds,
        "is_first_pick_slot" to isFirstPickSlot,
        "ally_preban_ids" to allyPrebanIds,
        "ally_preban_role_ids" to allyPrebanRoleIds,
        "ally_preban_element_ids" to allyPrebanElementIds,
        "enemy_preban_ids" to enemyPrebanIds,
        "enemy_preban_role_ids" to enemyPrebanRoleIds,
        "enemy_preban_element_ids" to enemyPrebanElementIds,
        "draft_filled_mask" to draftFilledMask,
        "candidate_mask" to candidateMask,
        "current_position_bucket_id" to currentPositionBucketId,
        "current_order" to currentOrder,
        "current_side_id" to currentSideId,
        "warfare_rule_id" to warfareRuleId,
    )
}
