# FangHao Learn

Static site for `learn.fanghao.top`.

## Local preview

```bash
npm run dev
```

Then open `http://localhost:4321`.

## GitHub Pages

This repo includes `CNAME`, so GitHub Pages will use `learn.fanghao.top` once DNS points there.

Cloudflare DNS record:

```txt
Type: CNAME
Name: learn
Target: FangWHao.github.io
Proxy: DNS only
```

## Vercel deploy

```bash
npx vercel --prod
```
