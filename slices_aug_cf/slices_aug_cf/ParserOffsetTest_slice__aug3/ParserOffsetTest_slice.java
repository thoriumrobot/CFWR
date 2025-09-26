/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        int __cfwr_entry76 = (false << (-529L * -274L));

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
      protected static long __cfwr_proc598(byte __cfwr_p0) {
        while (true) {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 7; __cfwr_i36++) {
            try {
            if (false && false) {
            while (true) {
            try {
            if (true || false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 3; __cfwr_i8++) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 10; __cfwr_i79++) {
            try {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 10; __cfwr_i14++) {
            if (true && true) {
            while ((-55.56 - -389L)) {
            try {
            while ((null | null)) {
            if (true || true) {
            try {
            while (('a' >> 118)) {
            while (true) {
            while (false) {
            while (true) {
            if ((-661L + ('p' - false)) || true) {
            try {
            if (((-863 * 61.92) * 61.07f) || true) {
            short __cfwr_temp37 = null;
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            try {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 4; __cfwr_i89++) {
            try {
            if (true && false) {
            if (true || (516 << 'P')) {
            if (((538 >> null) % 791L) && true) {
            if (((true ^ 'y') ^ (-300L % 59.37f)) && ((-15.29f & null) ^ -69.77f)) {
            try {
            return null;
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        return -82L;
    }
}
