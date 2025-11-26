import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fastf1
import pandas as pd
from rich.progress import track

# Clear fastf1 cache to avoid stale data
fastf1.Cache.clear_cache()

# Configure fastf1 logging (silence verbose logs)
fastf1.set_log_level(level=logging.ERROR)

# Configure application logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("f1-loader")

for name in ("fastf1", "fastf1.fastf1", "fastf1.api", "fastf1.req", "fastf1.core"):
    logging.getLogger(name).setLevel(logging.ERROR)

@dataclass
class Loader:
    """
    Loads F1 session results using fastf1 and saves them as csv files.

    Attributes
    ----------
    start : int
        Start year (inclusive).
    stop : int
        End year (inclusive).
    identifiers : list[str]
        Session identifiers (e.g. ["race", "sprint"]).
    output_dir : Path
        Directory where csv files will be written.
    sleep_between_years : float
        Pause (in seconds) between processing consecutive years.
    """

    start: int
    stop: int
    identifiers: List[str]
    output_dir: Path = Path("data/raw")
    sleep_between_years: float = 10.0

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loader initialized | start=%s stop=%s identifiers=%s output_dir=%s",
            self.start,
            self.stop,
            self.identifiers,
            self.output_dir,
        )

    def get_data(self, year: int, gp: int, identifier: str) -> pd.DataFrame:
        """
        Load the results of a specific session.

        Parameters
        ----------
        year : int
            Championship year.
        gp : int
            Round number.
        identifier : str
            Session identifier (e.g. "race", "sprint").

        Returns
        -------
        pd.DataFrame
            DataFrame with results. Returns an empty DataFrame if not available.
        """
        try:
            logger.debug(
                "Loading session | year=%s gp=%s identifier=%s",
                year,
                gp,
                identifier,
            )
            session = fastf1.get_session(year, gp, identifier)
            session.load(
                laps=False,
                telemetry=False,
                weather=False,
                messages=False,
            )

            df = session.results.copy()

            df["identifier"] = identifier
            df["date"] = str(session.date)
            df["year"] = session.date.year
            df["RoundNumber"] = int(session.event["RoundNumber"])
            df["Country"] = session.event["Country"]
            df["Location"] = session.event["Location"]
            df["OfficialEventName"] = session.event["OfficialEventName"]

            df = df.reset_index(drop=True)
            return df

        except ValueError as exc:
            logger.info(
                "Session unavailable (ValueError) | year=%s gp=%s identifier=%s | %s",
                year,
                gp,
                identifier,
                exc,
            )
            return pd.DataFrame()
        except Exception as exc:
            logger.error(
                "Unexpected error while loading session | year=%s gp=%s identifier=%s",
                year,
                gp,
                identifier,
                exc_info=exc,
            )
            return pd.DataFrame()

    def save_data(self, year: int, gp: int, identifier: str, df: pd.DataFrame) -> Path:
        """
        Save the DataFrame as a csv file in the configured output directory.

        Returns
        -------
        Path
            Full path of the saved file.
        """
        file_name = self.output_dir / f"{year}_{gp:02d}_{identifier}.csv"
        df.to_csv(file_name, sep=";", index=False)
        logger.info("File saved: %s (rows=%d)", file_name, len(df))
        return file_name

    def process_one(self, year: int, gp: int, identifier: str) -> bool:
        """
        Process a single session (load and save if there is data).

        Returns
        -------
        bool
            True if the session was found and saved; False otherwise.
        """
        df = self.get_data(year, gp, identifier)
        if df.empty:
            logger.info(
                "Session has no data | year=%s gp=%s identifier=%s",
                year,
                gp,
                identifier,
            )
            return False

        self.save_data(year, gp, identifier, df)
        return True

    def process_year(self, year: int) -> None:
        """
        Process all rounds of a given year for the configured session identifiers.

        If the main race ("race") is not available for a given round, it assumes
        the season has ended and stops processing that year.
        """
        logger.info("Processing year %s", year)
        try:
            df_rounds = fastf1.get_event_schedule(year)
        except Exception as exc:
            logger.error("Failed to get schedule for year %s", year, exc_info=exc)
            return

        year_rounds = int(df_rounds["RoundNumber"].max())
        logger.info("Year %s has %s rounds", year, year_rounds)

        rounds = range(1, year_rounds + 1)

        for gp in track(rounds, description=f"Processing rounds for {year}..."):
            logger.info("Processing round %s/%s for year %s", gp, year_rounds, year)
            result = False
            for identifier in self.identifiers:
                try:
                    result = self.process_one(year, gp, identifier)
                except ValueError:
                    pass

                # Keep original behavior: if there is no 'race', assume season ended.
                if not result and identifier == "race":
                    logger.info(
                        "Race not found for year=%s gp=%s. "
                        "Stopping processing for this year.",
                        year,
                        gp,
                    )
                    return

    def process_start_stop(self) -> None:
        """
        Process all years in the interval [start, stop].
        """
        years: Iterable[int] = range(self.start, self.stop + 1)

        for year in years:
            self.process_year(year)
            if year != self.stop and self.sleep_between_years > 0:
                logger.info(
                    "Sleeping %.1f seconds before processing the next year...",
                    self.sleep_between_years,
                )
                time.sleep(self.sleep_between_years)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download F1 session results using fastf1."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=2025,
        help="Start year (inclusive). Default: 2025",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=2025,
        help="End year (inclusive). Default: 2025",
    )
    parser.add_argument(
        "--identifiers",
        nargs="*",
        default=["race", "sprint"],
        help="List of session identifiers (e.g. race sprint quali).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where csv files will be saved.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=10.0,
        help="Pause (in seconds) between years. Default: 10",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = Loader(
        start=args.start,
        stop=args.stop,
        identifiers=args.identifiers,
        output_dir=args.output_dir,
        sleep_between_years=args.sleep,
    )

    loader.process_start_stop()


if __name__ == "__main__":
    main()
