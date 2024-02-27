"""Endpoints for Lilac Garden."""

from fastapi import APIRouter

from .garden_client import GardenEndpoint, list_endpoints
from .router_utils import RouteErrorHandler

router = APIRouter(route_class=RouteErrorHandler)


@router.get('/')
def get_available_endpoints() -> list[GardenEndpoint]:
  """Get the list of available sources."""
  return list_endpoints()
