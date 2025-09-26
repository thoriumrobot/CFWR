/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("unchecked")
  public static <K, V extends Record, R extends Record>
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
          Function<? super R, ? extends K> keyMapper) {
        Boolean __cfwr_result70 = null;


    return Collectors.groupingBy(
        keyMapper,
        LinkedHashMap::new,
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
      protected static Float __cfwr_aux977(Boolean __cfwr_p0, float __cfwr_p1, short __cfwr_p2) {
        Character __cfwr_entry4 = null;
        if ((null & (-559 ^ 62.39f)) || true) {
            while (true) {
            if (false || (67L - (76.84 / -59.70))) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 3; __cfwr_i74++) {
            short __cfwr_temp43 = (-230 / false);
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    static Integer __cfwr_temp853(Double __cfwr_p0) {
        try {
            while (true) {
            Character __cfwr_var17 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        return null;
    }
}
