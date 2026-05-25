# MCP Integration

Mintlify Hobby tier 自動提供 MCP endpoint，讓 Claude / Codex / 其他 agents 可以拉取手冊內容。

## Endpoint

部署後：`https://<sub-domain>.mintlify.app/mcp` （Streamable HTTP，無 auth）

## Claude Desktop / Claude Code 設定

```json
{
  "mcpServers": {
    "physics-gen-handbook": {
      "url": "https://<sub-domain>.mintlify.app/mcp"
    }
  }
}
```

## 與 sister handbooks 的協作

Pulsar / VLA-Handbook / Spatial-Handbook 都可以同時掛多個 MCP。建議：
- `physics-gen-handbook` 此倉
- `spatial-handbook` 對應 `kensou.mintlify.app/mcp`
- `vla-handbook`（待 deploy）

## Tools exposed

- `search` — 全 text 搜
- `get_page` — 拉 page markdown
- `list_pages` — list nav

具體 schema 看 Mintlify MCP docs。
