package com.e7.bphelper

import android.content.Context
import org.json.JSONObject

data class EncoderMetadata(
    val draftSlots: Int,
    val maxPreban: Int,
    val padHeroId: Int,
    val unkHeroId: Int,
    val allySideId: Int,
    val enemySideId: Int,
    val firstPickSlotIndices: IntArray,
    val sideOrderAllyFirst: IntArray,
    val sideOrderEnemyFirst: IntArray,
    val draftPositionBucketIds: IntArray,
    val heroToId: Map<String, Int>,
    val idToHero: List<String>,
    val heroRoleIds: IntArray,
    val heroElementIds: IntArray,
    val warfareRuleOrder: List<String>,
    val warfareRuleToId: Map<String, Int>,
    val warfareRulePriors: Map<String, Double>,
    val numHeroes: Int,
    val tfliteAssetPath: String,
    val requiresSelectTfOps: Boolean,
    val tfliteInputOrder: List<String>,
    val maskSpecialHeroes: Set<String>,
    val defaultTopK: Int,
    val positionBucketToId: Map<String, Int>,
    private val idToPositionBucket: Map<Int, String>,
) {
    fun positionBucketForId(bucketId: Int): String {
        return idToPositionBucket[bucketId] ?: "1"
    }
    fun heroId(code: String?): Int {
        if (code.isNullOrBlank()) {
            return padHeroId
        }
        return heroToId[code] ?: unkHeroId
    }

    fun roleIdForHeroId(heroId: Int): Int {
        if (heroId < 0 || heroId >= heroRoleIds.size) {
            return 0
        }
        return heroRoleIds[heroId]
    }

    fun elementIdForHeroId(heroId: Int): Int {
        if (heroId < 0 || heroId >= heroElementIds.size) {
            return 0
        }
        return heroElementIds[heroId]
    }

    fun warfareRuleId(ruleName: String): Int {
        return warfareRuleToId.getValue(ruleName)
    }

    companion object {
        private const val METADATA_ASSET_PATH = "data/transformer_encoder_metadata.json"

        fun load(context: Context): EncoderMetadata {
            val jsonText = context.assets.open(METADATA_ASSET_PATH).use { stream ->
                stream.readBytes().toString(Charsets.UTF_8)
            }
            val root = JSONObject(jsonText)
            val model = root.getJSONObject("model")
            val ranking = root.getJSONObject("ranking")

            val heroToIdJson = root.getJSONObject("hero_to_id")
            val heroToId = buildMap {
                val keys = heroToIdJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, heroToIdJson.getInt(key))
                }
            }

            val warfareRuleToIdJson = root.getJSONObject("warfare_rule_to_id")
            val warfareRuleToId = buildMap {
                val keys = warfareRuleToIdJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, warfareRuleToIdJson.getInt(key))
                }
            }

            val priorsJson = root.getJSONObject("warfare_rule_priors")
            val priors = buildMap {
                val keys = priorsJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, priorsJson.getDouble(key))
                }
            }

            val idToHero = root.getJSONArray("id_to_hero").let { array ->
                List(array.length()) { index -> array.getString(index) }
            }

            val warfareRuleOrder = root.getJSONArray("warfare_rule_order").let { array ->
                List(array.length()) { index -> array.getString(index) }
            }

            val maskSpecialHeroes = ranking.getJSONArray("mask_special_heroes").let { array ->
                buildSet {
                    for (index in 0 until array.length()) {
                        add(array.getString(index))
                    }
                }
            }

            val positionBucketToIdJson = root.getJSONObject("position_bucket_to_id")
            val positionBucketToId = buildMap {
                val keys = positionBucketToIdJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, positionBucketToIdJson.getInt(key))
                }
            }
            val idToPositionBucket = positionBucketToId.entries
                .filter { (bucket, _) -> bucket != "<PAD>" }
                .associate { (bucket, id) -> id to bucket }

            val tfliteInputOrder = model.optJSONArray("tflite_input_order")?.let { array ->
                List(array.length()) { index -> array.getString(index) }
            } ?: emptyList()

            return EncoderMetadata(
                draftSlots = root.getInt("draft_slots"),
                maxPreban = root.getInt("max_preban"),
                padHeroId = root.getInt("pad_hero_id"),
                unkHeroId = root.getInt("unk_hero_id"),
                allySideId = root.getInt("ally_side_id"),
                enemySideId = root.getInt("enemy_side_id"),
                firstPickSlotIndices = root.getJSONArray("first_pick_slot_indices").toIntArray(),
                sideOrderAllyFirst = root.getJSONArray("side_order_ally_first").toIntArray(),
                sideOrderEnemyFirst = root.getJSONArray("side_order_enemy_first").toIntArray(),
                draftPositionBucketIds = root.getJSONArray("draft_position_bucket_ids").toIntArray(),
                heroToId = heroToId,
                idToHero = idToHero,
                heroRoleIds = root.getJSONArray("hero_role_ids").toIntArray(),
                heroElementIds = root.getJSONArray("hero_element_ids").toIntArray(),
                warfareRuleOrder = warfareRuleOrder,
                warfareRuleToId = warfareRuleToId,
                warfareRulePriors = priors,
                numHeroes = model.getInt("num_heroes"),
                tfliteAssetPath = model.getString("tflite_asset_path"),
                requiresSelectTfOps = model.getBoolean("requires_select_tf_ops"),
                tfliteInputOrder = tfliteInputOrder,
                maskSpecialHeroes = maskSpecialHeroes,
                defaultTopK = ranking.getInt("default_top_k"),
                positionBucketToId = positionBucketToId,
                idToPositionBucket = idToPositionBucket,
            )
        }

        private fun org.json.JSONArray.toIntArray(): IntArray {
            return IntArray(length()) { index -> getInt(index) }
        }
    }
}
