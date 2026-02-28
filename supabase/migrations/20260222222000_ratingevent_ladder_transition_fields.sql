begin;

alter table public.ratingevent
    add column if not exists before_league text,
    add column if not exists before_division text,
    add column if not exists before_lp integer,
    add column if not exists after_league text,
    add column if not exists after_division text,
    add column if not exists after_lp integer,
    add column if not exists transition_type text not null default 'stable',
    add column if not exists major_promo_name text;

commit;
