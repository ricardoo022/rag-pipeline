-- KB schema para RAG (chunks de Markdown)
-- KISS: tabela única com vetor + FTS + filtros simples (empresa/cliente)

create extension if not exists vector;

-- Observação: evitamos dependências extras (uuid-ossp/pgcrypto).
-- Usamos chave surrogate BIGSERIAL e unicidade por (doc_path, chunk_ix).

create table if not exists public.kb_chunks (
  id bigserial primary key,
  doc_path text not null,
  chunk_ix int not null,
  content text not null,
  embedding vector(1536) not null,
  fts tsvector generated always as (to_tsvector('english', coalesce(content, ''))) stored,
  meta jsonb not null default '{}'::jsonb,
  -- filtros opcionais por tenant/empresa
  client_id uuid null,
  empresa text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (doc_path, chunk_ix)
);

create or replace function public.kb_touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists kb_chunks_set_updated_at on public.kb_chunks;
create trigger kb_chunks_set_updated_at
before update on public.kb_chunks
for each row execute function public.kb_touch_updated_at();

-- Corrective fix for databases created before `fts` switched from 'portuguese' to 'english'
-- (kb/ content is English; 'portuguese' barely stems it — see docs/markdown_header_drop.md).
-- `create table if not exists` above is a no-op on an existing table, so an already-created
-- `fts` column needs an explicit drop+recreate. Guarded to only run once: skips if `fts`
-- already uses 'english', so re-running `make migrate` doesn't churn on every apply.
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'kb_chunks' and column_name = 'fts'
      and generation_expression not ilike '%english%'
  ) then
    alter table public.kb_chunks drop column fts;
    alter table public.kb_chunks
      add column fts tsvector generated always as (to_tsvector('english', coalesce(content, ''))) stored;
  end if;
end $$;
