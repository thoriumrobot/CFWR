/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  static final long @MinLen(1) [] factorials = {
    1L,
    1L,
    1L * 2,
    1L * 2 * 3,
    1L * 2 * 3 * 4,
    1L * 2 * 3 * 4 * 5,
    1L * 2 * 3 * 4 * 5 * 6,
    1L * 2 * 3 * 4 * 5 * 6 * 7
  };

  static void binomialA(
      @NonNegative @LTLengthOf("Issue2494.factorials") int n,
      @NonNegative @LessThan("#1 + 1") int k) {
        boolean __cfwr_entry40 = false;

    @IndexFor("factorials") int j = k;
  }
    protected Object __cfwr_aux840() {
        return 61.98f;
        while ((-57.31 >> null)) {
            if ((-43.13 % 625) && ((-922 ^ null) << null)) {
            return (null ^ (352L << 538));
        }
            break; // Prevent infinite loops
        }
        while (((-0.16 ^ null) + (true - true))) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 6; __cfwr_i20++) {
            boolean __cfwr_elem20 = true;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
