/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        while (false) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 1; __cfwr_i88++) {
            return null;
        }
            break; // Prevent infinite loops
        }

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
      protected Float __cfwr_handle206(Object __cfwr_p0, String __cfwr_p1, Object __cfwr_p2) {
        while (false) {
            try {
            if (('U' + 50.80f) && (null % (76.82f / 'N'))) {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 8; __cfwr_i62++) {
            if (false && true) {
            if (true && false) {
            return null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
