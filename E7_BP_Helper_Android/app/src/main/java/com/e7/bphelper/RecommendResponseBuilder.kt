package com.e7.bphelper

import org.json.JSONArray
import org.json.JSONObject

object RecommendResponseBuilder {
    fun toJson(result: RecommendationResult): String {
        val payload = JSONObject()
        payload.put("phase", "pick")
        payload.put(
            "top_10_heroes",
            JSONArray().apply {
                result.topHeroes.forEach { hero ->
                    put(hero.heroCode)
                }
            },
        )
        payload.put(
            "top_10_rates",
            JSONArray().apply {
                result.topHeroes.forEach { hero ->
                    put(hero.rate)
                }
            },
        )
        return payload.toString()
    }
}
