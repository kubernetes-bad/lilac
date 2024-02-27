/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { GardenEndpoint } from '../models/GardenEndpoint';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class GardenService {

    /**
     * Get Available Endpoints
     * Get the list of available sources.
     * @returns GardenEndpoint Successful Response
     * @throws ApiError
     */
    public static getAvailableEndpoints(): CancelablePromise<Array<GardenEndpoint>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/garden/',
        });
    }

}
