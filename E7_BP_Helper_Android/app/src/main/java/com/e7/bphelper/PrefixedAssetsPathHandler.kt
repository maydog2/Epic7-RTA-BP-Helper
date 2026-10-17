package com.e7.bphelper

import android.content.Context
import android.webkit.MimeTypeMap
import android.webkit.WebResourceResponse
import androidx.webkit.WebViewAssetLoader
import java.io.IOException

class PrefixedAssetsPathHandler(
    private val context: Context,
    private val assetDirectory: String,
) : WebViewAssetLoader.PathHandler {
    override fun handle(path: String): WebResourceResponse? {
        val normalized = path.trimStart('/')
        val assetPath = if (assetDirectory.isEmpty()) {
            normalized
        } else {
            "$assetDirectory/$normalized"
        }
        return try {
            val inputStream = context.assets.open(assetPath)
            val extension = assetPath.substringAfterLast('.', "")
            val mimeType = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension) ?: "application/octet-stream"
            WebResourceResponse(mimeType, null, inputStream)
        } catch (_: IOException) {
            null
        }
    }
}
