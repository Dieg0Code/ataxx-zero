function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
}

function normalizeAssetPath(assetPath: string): string {
  return assetPath.replace(/^\/+/, "");
}

export function assetUrl(assetPath: string): string {
  const baseUrl = normalizeBaseUrl(import.meta.env.BASE_URL ?? "/");
  return `${baseUrl}${normalizeAssetPath(assetPath)}`;
}
