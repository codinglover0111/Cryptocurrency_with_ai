import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { createMiddlewareClient } from "@supabase/auth-helpers-nextjs";

const ADMIN_PATHS = ["/admin"];

export async function middleware(req: NextRequest) {
  const res = NextResponse.next();
  const supabase = createMiddlewareClient({ req, res });
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const path = req.nextUrl.pathname;
  const isAdminPath = ADMIN_PATHS.some((p) => path === p || path.startsWith(`${p}/`));

  if (!isAdminPath) {
    return res;
  }

  if (!session) {
    const redirectUrl = new URL("/login", req.nextUrl.origin);
    redirectUrl.searchParams.set("redirectedFrom", path);
    return NextResponse.redirect(redirectUrl);
  }

  const role =
    (session.user.app_metadata as Record<string, unknown>)?.role ??
    (session.user.user_metadata as Record<string, unknown>)?.role;
  if (role !== "admin") {
    return NextResponse.redirect(new URL("/", req.nextUrl.origin));
  }

  return res;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
