/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("unchecked")
  public static <K, V extends Record, R extends Record>
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
          Function<? super R, ? extends K> keyMapper) {
        Boolean __cfwr_val74 = null;


    return Collectors.groupingBy(
        keyMapper,
        LinkedHashMap::new,
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
      static short __cfwr_proc469(Double __cfwr_p0, long __cfwr_p1) {
        return 99.60;
        if (true || false) {
            if (false && true) {
            try {
            return (null / (-30.51 & 495));
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
        try {
            if (true && false) {
            try {
            int __cfwr_data18 = 243;
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        return null;
    }
}
