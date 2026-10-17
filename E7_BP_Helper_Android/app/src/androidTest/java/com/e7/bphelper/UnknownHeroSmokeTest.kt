package com.e7.bphelper

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class UnknownHeroSmokeTest {
    @Test
    fun parserAcceptsHeroOutsideTransformerVocabulary() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val metadata = EncoderMetadata.load(context)
        val parser = RecommendRequestParser(metadata)

        val uri = Uri.parse(
            "https://appassets.androidplatform.net/api/recommend" +
                "?user_picks=c4065" +
                "&enemy_picks=" +
                "&ally_preban=" +
                "&enemy_preban=" +
                "&first_pick_team=My%20Team" +
                "&warfare_rules=ANY",
        )

        val input = parser.parse(uri)
        assertEquals(listOf("c4065"), input.userPicks)
    }

    @Test
    fun inferenceBuilderNormalizesUnknownHeroToUnkToken() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val metadata = EncoderMetadata.load(context)
        val playable = metadata.heroToId.keys
            .filter { it != "<PAD>" && it != "<UNK>" }
            .toSet()

        assertFalse("c4065 should be absent from transformer vocab", "c4065" in playable)

        val builder = InferenceArrayBuilder(metadata)
        val tensors = builder.build(
            DraftRecommendInput(
                userPicks = listOf("c4065"),
                enemyPicks = emptyList(),
                allyPreban = emptyList(),
                enemyPreban = emptyList(),
                firstPickTeam = "My Team",
                warfareRules = "ANY",
            ),
        )

        assertEquals(metadata.padHeroId, tensors.draftHeroIds[0])
    }
}
