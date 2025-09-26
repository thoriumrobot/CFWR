/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("unchecked")
  public static <K, V extends Record, R extends Record>
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
          Function<? super R, ? extends K> keyMapper) {
        return null;


    return Collectors.groupingBy(
        keyMapper,
        LinkedHashMap::new,
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
      public int __cfwr_func199(Boolean __cfwr_p0
        Integer __cfwr_item98 = null;
, boolean __cfwr_p1) {
        return true;
        while ((null & -508)) {
            boolean __cfwr_node3 = true;
            break; // Prevent infinite loops
        }
        return -791;
    }
}
