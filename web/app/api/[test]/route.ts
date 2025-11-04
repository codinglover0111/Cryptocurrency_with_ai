import { NextRequest, NextResponse } from "next/server";

type RouteContext = {
  params: { test?: string };
};

export async function GET(_request: NextRequest, context: RouteContext) {
  const test =
    context.params?.test ??
    _request.nextUrl.pathname.split("/").filter(Boolean).at(-1);

  if (!test) {
    return NextResponse.json(
      { error: "missing dynamic segment" },
      { status: 400 }
    );
  }

  return NextResponse.json({ value: test });
}
