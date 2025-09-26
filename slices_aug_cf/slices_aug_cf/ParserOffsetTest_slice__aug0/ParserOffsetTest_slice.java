/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        return null;

    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
      @IndexFor("a") int k = i;
      // :: error: (assignment)
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
      @LTLengthOf("a") int k3 = j;
    }
      Boolean __cfwr_proc408(int __cfwr_p0, Double __cfwr_p1, Boolean __cfwr_p2) {
        Float __cfwr_entry89 = null;
        try {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 3; __cfwr_i90++) {
            Integer __cfwr_val95 = null;
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        for (int __cfwr_i19 = 0; __cfwr_i19 < 7; __cfwr_i19++) {
            int __cfwr_elem2 = (-680 & (-410L / -41.43f));
        }
        return null;
    }
    private static Object __cfwr_util757(double __cfwr_p0, int __cfwr_p1, Integer __cfwr_p2) {
        boolean __cfwr_data48 = false;
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
