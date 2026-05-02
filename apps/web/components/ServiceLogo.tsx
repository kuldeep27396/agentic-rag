import Image from "next/image";

const icons: Record<string, string> = {
  nextjs: "nextdotjs",
  react: "react",
  typescript: "typescript",
  tailwind: "tailwindcss",
  fastapi: "fastapi",
  python: "python",
  pydantic: "pydantic",
  httpx: "python",
  vercel: "vercel",
  upstash: "upstash",
  qstash: "upstash",
  milvus: "milvus",
  openrouter: "openrouter",
  pypdf: "apachepdfbox",
  blob: "vercel",
  gemini: "googlegemini",
};

const labels: Record<string, string> = {
  httpx: "httpx",
  qstash: "QStash",
  pypdf: "pypdf",
  blob: "Blob",
  gemini: "Gemma 4",
};

const sizes: Record<string, number> = {
  TS: 20,
  httpx: 20,
  qstash: 20,
  pypdf: 20,
  blob: 20,
  gemini: 20,
};

function ServiceLogo({ name, size = 24 }: { name: string; size?: number }) {
  const slug = icons[name];
  const label = labels[name] ?? name;
  const s = sizes[name] ?? size;

  if (!slug) {
    return (
      <span className="inline-flex items-center justify-center rounded-lg bg-muted px-2 py-1 text-[10px] font-bold text-muted-foreground">
        {name}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center justify-center rounded-lg bg-muted/60 px-2 py-1 gap-1.5">
      <Image
        src={`https://cdn.simpleicons.org/${slug}`}
        alt={label}
        width={s}
        height={s}
        unoptimized
      />
      {name !== "blob" && name !== "TS" && (
        <span className="text-[10px] font-semibold text-muted-foreground leading-none">{label}</span>
      )}
    </span>
  );
}

export default ServiceLogo;
