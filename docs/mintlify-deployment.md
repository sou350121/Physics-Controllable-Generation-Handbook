# Mintlify Deployment

跟 Spatial-Intelligence-Handbook 同套（Hobby tier, free）。

## 步驟

1. **Mintlify dashboard**：用 GitHub OAuth 登入，「Connect repository」選 `sou350121/Physics-Controllable-Generation-Handbook`
2. **設 docs.json**：本倉已備（在 root）— navigation 已寫好
3. **首次 build**：dashboard 點 "Update"；之後 push 會 7s incremental rebuild
4. **自訂 sub-domain**：例如 `physics-gen-kensou.mintlify.app`

## 注意

- README.md 會被 Mintlify nav 隱藏（與 spatial 行為一致）
- 新增 dissection 要去 dashboard 點 Manual Update — incremental rebuild 不抓新文件
- 內鏈用 root-relative：`/foundations/...` 不要寫成 `../foundations/...`
- 標題避免 em dash 與 apostrophe（會破 anchor）

## 與 Spatial 共用 quirks

詳見 spatial-handbook 已記錄的 mintlify-gotchas（同套）：
- `docs.json` 中 pages 路徑不含 `.md` 後綴
- 隱藏路徑：以 `_` 開頭的目錄會被略過
- 多 line code block 在 nav 中無效

## MCP endpoint

Hobby tier 自帶 `/mcp` Streamable HTTP endpoint，無 auth。
寫到 README + AGENTS.md 後讓 sister agents 可拉。
