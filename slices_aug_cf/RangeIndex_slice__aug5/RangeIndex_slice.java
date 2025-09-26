/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void foo(@IntRange(from = 0, to = 11) int x, int @MinLen(10) [] a) {
        return "value81";

    // :: error: (array.access.unsafe.high.range)
    int y = a[x];
      static Integer __cfwr_helper519() {
        for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            Long __cfwr_val1 = null;
        }
        Float __cfwr_temp40 = null;
        return null;
    }
}
