import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabaseServer";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://localhost:8000";

async function forward(req: NextRequest, method: string) {
  const supabase = createSupabaseServerClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    return NextResponse.json(
      { error: "Unauthorized" },
      {
        status: 401,
      },
    );
  }

  const path = req.nextUrl.pathname.replace("/api/proxy", "");
  const targetUrl = new URL(path + req.nextUrl.search, FASTAPI_BASE_URL);

  const init: RequestInit = {
    method,
    headers: {
      Authorization: `Bearer ${session.access_token}`,
    },
  };

  // Pass JSON body if present
  if (method !== "GET" && method !== "HEAD") {
    const bodyText = await req.text();
    if (bodyText) {
      init.body = bodyText;
      init.headers = {
        ...init.headers,
        "Content-Type": req.headers.get("content-type") || "application/json",
      };
    }
  }

  // Forward select headers for CORS/debug
  const incomingHeaders = ["x-request-id", "x-trace-id"];
  incomingHeaders.forEach((key) => {
    const value = req.headers.get(key);
    if (value) {
      (init.headers as Record<string, string>)[key] = value;
    }
  });

  const resp = await fetch(targetUrl.toString(), init);
  const respBody = await resp.text();
  return new NextResponse(respBody, {
    status: resp.status,
    headers: {
      "content-type": resp.headers.get("content-type") || "application/json",
    },
  });
}

export const GET = (req: NextRequest) => forward(req, "GET");
export const POST = (req: NextRequest) => forward(req, "POST");
export const PUT = (req: NextRequest) => forward(req, "PUT");
export const PATCH = (req: NextRequest) => forward(req, "PATCH");
export const DELETE = (req: NextRequest) => forward(req, "DELETE");
