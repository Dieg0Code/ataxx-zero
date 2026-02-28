begin;

alter table public.game
    add column if not exists created_by_user_id uuid null references public."user"(id);

create index if not exists ix_game_created_by_user_id
    on public.game (created_by_user_id);

commit;
