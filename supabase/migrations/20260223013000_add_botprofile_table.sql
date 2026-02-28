begin;

create table if not exists public.botprofile (
    user_id uuid not null,
    agent_type public.agenttype not null,
    heuristic_level varchar(16),
    model_mode varchar(16),
    enabled boolean not null default true,
    created_at timestamp without time zone not null default timezone('utc', now()),
    updated_at timestamp without time zone not null default timezone('utc', now()),
    constraint pk_botprofile primary key (user_id),
    constraint fk_botprofile_user_id foreign key (user_id) references public."user"(id) on delete cascade,
    constraint uq_bot_profile_user_id unique (user_id)
);

create index if not exists ix_botprofile_agent_type on public.botprofile (agent_type);
create index if not exists ix_botprofile_enabled on public.botprofile (enabled);

commit;
