package com.e7.bphelper

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

class TfliteEngineDebugRunner(
    private val context: Context,
) {
    fun runAndLog(tag: String = TAG): DebugRunResult {
        val metadata = EncoderMetadata.load(context)
        val debugCase = loadDebugCase()
        val input = DraftRecommendInput(
            userPicks = debugCase.userPicks,
            enemyPicks = debugCase.enemyPicks,
            allyPreban = debugCase.allyPreban,
            enemyPreban = debugCase.enemyPreban,
            firstPickTeam = debugCase.firstPickTeam,
            warfareRules = debugCase.warfareRules,
            topK = debugCase.topK,
        )

        TfliteRecommendEngine(context, metadata).use { engine ->
            val result = engine.predict(input)
            val actualHeroes = result.topHeroes.map { it.heroCode }
            val actualRates = result.topHeroes.map { it.rate }
            val heroesMatch = actualHeroes == debugCase.expectedTopHeroes
            val ratesClose = compareRates(actualRates, debugCase.expectedTopRates)

            Log.i(tag, "Phase 3A debug case: ${debugCase.id}")
            Log.i(tag, "Actual top heroes: ${actualHeroes.joinToString()}")
            Log.i(tag, "Expected top heroes: ${debugCase.expectedTopHeroes.joinToString()}")
            Log.i(tag, "Top heroes match: $heroesMatch")
            Log.i(tag, "Top rates close (<=0.01 max diff): $ratesClose")
            result.topHeroes.forEachIndexed { index, hero ->
                Log.i(
                    tag,
                    "  #${index + 1} ${hero.heroCode} ${"%.4f".format(hero.rate)}%",
                )
            }

            return DebugRunResult(
                caseId = debugCase.id,
                heroesMatch = heroesMatch,
                ratesClose = ratesClose,
                actualHeroes = actualHeroes,
                expectedHeroes = debugCase.expectedTopHeroes,
            )
        }
    }

    private fun compareRates(actual: List<Double>, expected: List<Double>): Boolean {
        if (actual.size != expected.size) {
            return false
        }
        return actual.zip(expected).all { (left, right) ->
            kotlin.math.abs(left - right) <= 0.01
        }
    }

    private fun loadDebugCase(): DebugCase {
        val jsonText = context.assets.open(DEBUG_CASE_ASSET_PATH).use { stream ->
            stream.readBytes().toString(Charsets.UTF_8)
        }
        val root = JSONObject(jsonText)
        val input = root.getJSONObject("input")
        return DebugCase(
            id = root.getString("id"),
            userPicks = input.getStringList("user_picks"),
            enemyPicks = input.getStringList("enemy_picks"),
            allyPreban = input.getStringList("ally_preban"),
            enemyPreban = input.getStringList("enemy_preban"),
            firstPickTeam = input.getString("first_pick_team"),
            warfareRules = input.getString("warfare_rules"),
            topK = input.optInt("top_k", 10),
            expectedTopHeroes = root.getStringList("expected_top_heroes"),
            expectedTopRates = root.getDoubleList("expected_top_rates"),
        )
    }

    private fun JSONObject.getStringList(key: String): List<String> {
        val array = optJSONArray(key) ?: JSONArray()
        return List(array.length()) { index -> array.getString(index) }
    }

    private fun JSONObject.getDoubleList(key: String): List<Double> {
        val array = getJSONArray(key)
        return List(array.length()) { index -> array.getDouble(index) }
    }

    data class DebugCase(
        val id: String,
        val userPicks: List<String>,
        val enemyPicks: List<String>,
        val allyPreban: List<String>,
        val enemyPreban: List<String>,
        val firstPickTeam: String,
        val warfareRules: String,
        val topK: Int,
        val expectedTopHeroes: List<String>,
        val expectedTopRates: List<Double>,
    )

    data class DebugRunResult(
        val caseId: String,
        val heroesMatch: Boolean,
        val ratesClose: Boolean,
        val actualHeroes: List<String>,
        val expectedHeroes: List<String>,
    )

    companion object {
        private const val TAG = "TfliteEngineDebug"
        private const val DEBUG_CASE_ASSET_PATH = "data/android_tflite_debug_case.json"
    }
}
