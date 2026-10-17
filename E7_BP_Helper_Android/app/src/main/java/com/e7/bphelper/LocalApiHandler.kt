package com.e7.bphelper

import android.content.Context
import android.net.Uri
import android.util.Log
import android.webkit.WebResourceResponse
import org.json.JSONObject
import java.io.ByteArrayInputStream

class LocalApiHandler(context: Context) {
    private val appContext = context.applicationContext
    private val heroesJson: String by lazy { loadHeroesJson() }
    private val metadata: EncoderMetadata by lazy { EncoderMetadata.load(appContext) }
    private val artifacts: RecommendationArtifacts by lazy { RecommendationArtifacts(appContext) }
    private val requestParser: RecommendRequestParser by lazy { RecommendRequestParser(metadata) }
    private val prebanRecommender: PrebanRecommender by lazy { PrebanRecommender(artifacts) }
    private val recommendEngine: TfliteRecommendEngine by lazy {
        TfliteRecommendEngine(appContext, metadata)
    }
    private val recommendationEngine: LocalRecommendationEngine by lazy {
        LocalRecommendationEngine(metadata, recommendEngine, artifacts)
    }
    private val recommendEngineLock = Any()

    fun handle(uri: Uri): WebResourceResponse? {
        val path = uri.path ?: return null
        return when {
            path == "/api/status" || path == "/status" -> jsonResponse(200, statusPayload())
            path == "/api/heroes" -> jsonResponse(200, heroesJson)
            path.startsWith("/api/heroes/") && path.endsWith("/avatar") -> avatarResponse(path)
            path.startsWith("/api/icons/elements/") -> iconResponse(path, "CharacterUI/elements")
            path.startsWith("/api/icons/roles/") -> iconResponse(path, "CharacterUI/roles")
            path == "/api/recommend" || path == "/recommend" -> recommendResponse(uri)
            path == "/api/preban_recommend" || path == "/preban_recommend" -> prebanResponse(uri)
            else -> null
        }
    }

    private fun recommendResponse(uri: Uri): WebResourceResponse {
        return try {
            Log.i(RECOMMEND_TAG, "Parsing /api/recommend request: ${uri.query}")
            val input = requestParser.parse(uri)
            Log.i(
                RECOMMEND_TAG,
                "Draft input user=${input.userPicks} enemy=${input.enemyPicks} " +
                    "allyPreban=${input.allyPreban} enemyPreban=${input.enemyPreban} " +
                    "firstPick=${input.firstPickTeam} warfare=${input.warfareRules}",
            )

            val payload = synchronized(recommendEngineLock) {
                recommendationEngine.recommend(input)
            }
            val topHeroes = payload.optJSONArray("top_10_heroes")?.let { array ->
                List(array.length()) { index -> array.getString(index) }
            }.orEmpty()
            Log.i(RECOMMEND_TAG, "Returning phase=${payload.optString("phase")} top=${topHeroes.joinToString()}")
            jsonResponse(200, payload.toString())
        } catch (error: RecommendRequestException) {
            Log.w(RECOMMEND_TAG, "Invalid /api/recommend request: ${error.message}")
            jsonResponse(
                400,
                JSONObject().put("message", error.message ?: "Invalid recommendation request").toString(),
            )
        } catch (error: Exception) {
            Log.e(RECOMMEND_TAG, "Recommendation failed", error)
            jsonResponse(
                500,
                JSONObject()
                    .put("message", error.message ?: "Error generating recommendation")
                    .toString(),
            )
        }
    }

    private fun prebanResponse(uri: Uri): WebResourceResponse {
        return try {
            val excludedHeroes = parseCsv(uri.getQueryParameter("excluded_heroes"))
            val topK = prebanRecommender.parseTopK(uri.getQueryParameter("top_k"))
            val source = prebanRecommender.resolvePrebanSource(uri.getQueryParameter("preban_side"))
            val firstPickSide = prebanRecommender.resolvePrebanFirstPickSide(
                userFirstPickSide = DraftPositionUtils.parseFirstPickSide(
                    uri.getQueryParameter("first_pick_team"),
                ),
                prebanSource = source,
            )
            val payload = prebanRecommender.recommend(
                excludedHeroes = excludedHeroes,
                topK = topK,
                source = source,
                firstPickSide = firstPickSide,
            )
            jsonResponse(200, payload.toString())
        } catch (error: Exception) {
            Log.e(RECOMMEND_TAG, "Preban recommendation failed", error)
            jsonResponse(
                500,
                JSONObject()
                    .put("message", error.message ?: "Error generating preban recommendation")
                    .toString(),
            )
        }
    }

    private fun parseCsv(rawValue: String?): List<String> {
        if (rawValue.isNullOrBlank()) {
            return emptyList()
        }
        return rawValue.split(",").map { it.trim() }.filter { it.isNotEmpty() }
    }

    private fun statusPayload(): String {
        val payload = JSONObject()
        payload.put("message", "Server is running")
        payload.put("app", "e7_bp_helper")
        payload.put("frontend_enabled", true)
        payload.put("reranker_enabled", true)
        payload.put("debug_recommendations_enabled", false)
        payload.put("local_recommend_enabled", true)
        return payload.toString()
    }

    private fun loadHeroesJson(): String {
        return appContext.assets.open("data/hero_details.json").use { stream ->
            stream.readBytes().toString(Charsets.UTF_8)
        }
    }

    private fun avatarResponse(path: String): WebResourceResponse? {
        val segments = path.trim('/').split('/')
        if (segments.size < 4) {
            return null
        }
        val heroCode = segments[2]
        return assetBinaryResponse("dataset/$heroCode/c.png", "image/png")
    }

    private fun iconResponse(path: String, assetDir: String): WebResourceResponse? {
        val fileName = path.substringAfterLast('/')
        if (fileName.isBlank() || fileName.contains("..")) {
            return null
        }
        return assetBinaryResponse("$assetDir/$fileName", "image/png")
    }

    private fun assetBinaryResponse(assetPath: String, mimeType: String): WebResourceResponse? {
        return try {
            val bytes = appContext.assets.open(assetPath).use { it.readBytes() }
            WebResourceResponse(
                mimeType,
                null,
                200,
                "OK",
                emptyMap(),
                ByteArrayInputStream(bytes),
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun jsonResponse(statusCode: Int, body: String): WebResourceResponse {
        val statusText = when (statusCode) {
            200 -> "OK"
            400 -> "Bad Request"
            500 -> "Internal Server Error"
            501 -> "Not Implemented"
            else -> "Error"
        }
        return WebResourceResponse(
            "application/json",
            "utf-8",
            statusCode,
            statusText,
            mapOf("Access-Control-Allow-Origin" to "*"),
            ByteArrayInputStream(body.toByteArray(Charsets.UTF_8)),
        )
    }

    companion object {
        private const val RECOMMEND_TAG = "LocalApiRecommend"
    }
}
