/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        if (true || false) {
            while ((null & (186 - null))) {
            while (true) {
            String __cfwr_obj18 = "temp32";
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
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
      private static boolean __cfwr_helper977(Integer __cfwr_p0, short __cfwr_p1, byte __cfwr_p2) {
        byte __cfwr_temp33 = null;
        return ((299L * 774L) >> true);
    }
    public static long __cfwr_proc5() {
        while (false) {
            while (false) {
            try {
            return -306L;
        } catch (Exception __cfwr_e86) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        if (true || true) {
            return null;
        }
        int __cfwr_data6 = -428;
        return (-83 * true);
    }
}
