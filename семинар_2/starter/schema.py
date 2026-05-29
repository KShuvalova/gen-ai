from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


CURRENT_YEAR = datetime.now().year

CITIES = (
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Ростов-на-Дону",
    "Краснодар",
    "Пермь",
)

SPECIALITIES = (
    "HR-специалист",
    "бухгалтер",
    "маркетолог",
    "менеджер по продажам",
    "учитель",
    "инженер",
    "медицинская сестра",
    "юрист",
    "аналитик данных",
    "специалист по закупкам",
)

DESIRED_COURSES = (
    "Управление проектами",
    "Data Analytics для бизнеса",
    "Цифровой маркетинг",
    "HR-аналитика",
    "Финансовый учет и Excel",
    "Педагогический дизайн",
    "Охрана труда",
    "Бизнес-аналитика и BPMN",
)


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=60)

    @field_validator("city")
    @classmethod
    def city_must_be_from_list(cls, value: str) -> str:
        if value not in CITIES:
            raise ValueError(f"Город «{value}» не входит в утвержденный список")
        return value


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=90)
    age: int = Field(ge=22, le=65)
    address: Address

    speciality: Literal[
        "HR-специалист",
        "бухгалтер",
        "маркетолог",
        "менеджер по продажам",
        "учитель",
        "инженер",
        "медицинская сестра",
        "юрист",
        "аналитик данных",
        "специалист по закупкам",
    ]

    desired_course: Literal[
        "Управление проектами",
        "Data Analytics для бизнеса",
        "Цифровой маркетинг",
        "HR-аналитика",
        "Финансовый учет и Excel",
        "Педагогический дизайн",
        "Охрана труда",
        "Бизнес-аналитика и BPMN",
    ]

    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_year_not_in_future(cls, value: int) -> int:
        if value > CURRENT_YEAR:
            raise ValueError("Год окончания не может быть позже текущего года")
        return value

    @model_validator(mode="after")
    def check_age_education_and_experience_consistency(self) -> "Application":
        # graduation_year + 22 <= CURRENT_YEAR + age
        # То есть возраст и год окончания не должны противоречить друг другу.
        if self.graduation_year + 22 > CURRENT_YEAR + self.age:
            raise ValueError(
                "Возраст и год окончания противоречат друг другу: "
                "на момент окончания должно быть не меньше 22 лет"
            )

        max_possible_experience = min(
            40,
            self.age - 22,
            CURRENT_YEAR - self.graduation_year + 1,
        )

        if self.years_of_experience > max_possible_experience:
            raise ValueError(
                "Стаж слишком большой для указанного возраста и года окончания"
            )

        return self
