package com.e7.bphelper

data class InferenceTensors(
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
    val candidateMask: FloatArray,
    val currentPositionBucketId: Int,
    val currentOrder: Int,
    val currentSideId: Int,
    val warfareRuleId: Int,
) {
    val draftFilledMask: FloatArray
        get() = FloatArray(draftHeroIds.size) { index ->
            if (draftHeroIds[index] > 0) 1f else 0f
        }

    fun withWarfareRuleId(ruleId: Int): InferenceTensors = copy(warfareRuleId = ruleId)
}
