/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("unchecked")
  public static <K, V extends Record, R extends Record>
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
          Function<? super R, ? extends K> keyMapper) {
        Character __cfwr_entry86 = null;


    return Collectors.groupingBy(
        keyMapper,
        LinkedHashMap::new,
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0
        try {
            byte __cfwr_node49 = (null & 619L);
        } catch (Exception __cfwr_e64) {
            // ignore
        }
]));
      public Float __cfwr_util414() {
        try {
            try {
            short __cfwr_result19 = (false / 'd');
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        return null;
    }
}
