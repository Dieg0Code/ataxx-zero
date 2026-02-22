from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import text

# Permite ejecutar el script desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


SeasonRow = tuple[UUID, str, str, bool]


async def ensure_active_season(name: str) -> SeasonRow:
    from api.db.session import get_sessionmaker

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        # Desactivar temporadas activas previas para mantener una sola activa.
        await session.execute(text("UPDATE season SET is_active = FALSE WHERE is_active = TRUE"))
        await session.commit()

        existing = await session.execute(
            text(
                """
                SELECT id, name, starts_at::text, is_active
                FROM season
                WHERE name = :name
                LIMIT 1
                """
            ),
            {"name": name},
        )
        existing_row = existing.first()

        if existing_row is None:
            season_id = str(uuid4())
            created = await session.execute(
                text(
                    """
                    INSERT INTO season (id, name, starts_at, ends_at, is_active, created_at)
                    VALUES (
                        :id,
                        :name,
                        (now() at time zone 'utc'),
                        NULL,
                        TRUE,
                        (now() at time zone 'utc')
                    )
                    RETURNING id, name, starts_at::text, is_active
                    """
                ),
                {"id": season_id, "name": name},
            )
            await session.commit()
            row = created.one()
            return (row[0], row[1], row[2], row[3])

        updated = await session.execute(
            text(
                """
                UPDATE season
                SET is_active = TRUE,
                    ends_at = NULL
                WHERE id = :id
                RETURNING id, name, starts_at::text, is_active
                """
            ),
            {"id": str(existing_row[0])},
        )
        await session.commit()
        row = updated.one()
        return (row[0], row[1], row[2], row[3])


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Crea o activa una temporada de ranking.")
    parser.add_argument(
        "--name",
        default=f"Season {datetime.now().strftime('%Y-%m-%d')}",
        help="Nombre de la temporada a activar/crear.",
    )
    args = parser.parse_args()

    season = await ensure_active_season(args.name)
    print("Active season ready:")
    print(f"  id={season[0]}")
    print(f"  name={season[1]}")
    print(f"  starts_at={season[2]}")
    print(f"  is_active={season[3]}")

    # Cierra el engine para evitar ruido de transporte SSL al apagar el loop en Windows.
    from api.db.session import get_engine

    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main_async())
