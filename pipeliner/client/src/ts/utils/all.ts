type Extracted<T> = T extends PromiseLike<infer U> ? U : T;
type Values<T extends unknown[]> = T extends []
    ? []
    : T extends [infer U, ...infer R]
    ? [Extracted<U>, ...Values<R>]
    : never;

export function all<T extends unknown[]>(...values: T) {
    return Promise.all(values) as Promise<Values<T>>;
}
