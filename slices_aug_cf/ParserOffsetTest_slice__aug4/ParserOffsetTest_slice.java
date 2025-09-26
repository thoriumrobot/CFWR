/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        while (((null * true) % true)) {
            while (false) {
            byte __cfwr_node86 = (332L % true);
            break; // Prevent infinite loops
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
      static boolean __cfwr_process814(double __cfwr_p0, float __cfwr_p1, boolean __cfwr_p2) {
        for (int __cfwr_i19 = 0; __cfwr_i19 < 5; __cfwr_i19++) {
            try {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 1; __cfwr_i98++) {
            if (true || true) {
            try {
            Double __cfwr_val15 = null;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        for (int __cfwr_i1 = 0; __cfwr_i1 < 10; __cfwr_i1++) {
            while (false) {
            return (87.50f ^ 439L);
            break; // Prevent infinite loops
        }
        }
        return false;
    }
}
