import { createClientComponentClient } from "@supabase/auth-helpers-nextjs";

export const createSupabaseBrowserClient = () =>
  createClientComponentClient({
    options: {
      global: {
        fetch,
      },
    },
  });
