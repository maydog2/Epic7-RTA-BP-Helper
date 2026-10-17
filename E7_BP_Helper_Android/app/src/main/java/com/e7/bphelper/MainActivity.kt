package com.e7.bphelper

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import androidx.webkit.WebViewAssetLoader

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private lateinit var assetLoader: WebViewAssetLoader
    private lateinit var localApiHandler: LocalApiHandler

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        localApiHandler = LocalApiHandler(this)
        assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/public/", PrefixedAssetsPathHandler(this, "public"))
            .addPathHandler("/assets/", PrefixedAssetsPathHandler(this, "public/assets"))
            .addPathHandler("/dataset/", PrefixedAssetsPathHandler(this, "dataset"))
            .addPathHandler("/CharacterUI/", PrefixedAssetsPathHandler(this, "CharacterUI"))
            .build()

        webView = WebView(this)
        setContentView(webView)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            cacheMode = WebSettings.LOAD_DEFAULT
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest,
            ): WebResourceResponse? {
                val uri = request.url
                val path = uri.path.orEmpty()
                if (path.startsWith("/api/") || path == "/status") {
                    localApiHandler.handle(uri)?.let { return it }
                }
                return assetLoader.shouldInterceptRequest(uri)
            }
        }

        webView.loadUrl("https://appassets.androidplatform.net/public/index.html")

        Thread {
            try {
                val result = TfliteEngineDebugRunner(this).runAndLog()
                if (!result.heroesMatch || !result.ratesClose) {
                    Log.w(
                        "TfliteEngineDebug",
                        "Phase 3A parity mismatch for ${result.caseId}",
                    )
                }
            } catch (error: Exception) {
                Log.e("TfliteEngineDebug", "Phase 3A debug run failed", error)
            }
        }.start()
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
