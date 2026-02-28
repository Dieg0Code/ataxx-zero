begin;

do $$
begin
    if not exists (
        select 1
        from pg_type t
        join pg_namespace n on n.oid = t.typnamespace
        where t.typname = 'queueentrystatus' and n.nspname = 'public'
    ) then
        create type public.queueentrystatus as enum ('WAITING', 'MATCHED', 'CANCELED');
    end if;
end $$;

create table if not exists public.queueentry (
    id uuid not null primary key,
    season_id uuid not null references public.season(id),
    user_id uuid not null references public."user"(id),
    rating_snapshot double precision not null,
    status public.queueentrystatus not null default 'WAITING',
    matched_game_id uuid null references public.game(id),
    created_at timestamp without time zone not null default timezone('utc', now()),
    updated_at timestamp without time zone not null default timezone('utc', now()),
    matched_at timestamp without time zone null,
    canceled_at timestamp without time zone null,
    constraint uq_queueentry_season_user unique (season_id, user_id)
);

create index if not exists ix_queueentry_season_id on public.queueentry (season_id);
create index if not exists ix_queueentry_user_id on public.queueentry (user_id);
create index if not exists ix_queueentry_status on public.queueentry (status);
create index if not exists ix_queueentry_matched_game_id on public.queueentry (matched_game_id);

commit;
