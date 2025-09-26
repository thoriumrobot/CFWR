/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("unchecked")
  public static <K, V extends Record, R extends Record>
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
          Function<? super R, ? extends K> keyMapper) {
        try {
            return null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }


    return Collectors.groupingBy(
        keyMapper,
        LinkedHashMap::new,
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
      static Object __cfwr_process257(double __cfwr_p0) {
        return 'M';
        if (true || (-64.31 + -430)) {
            float __cfwr_result65 = 15.43f;
        }
        return null;
        return null;
    }
}
