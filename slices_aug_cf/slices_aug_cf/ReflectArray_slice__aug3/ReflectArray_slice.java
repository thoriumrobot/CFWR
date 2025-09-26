/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        if (true && true) {
            short __cfwr_entry65 = null;
        }

    /
        return null;
/ :: error: (argument)
    Array.newInstance(Object.class, i);
    if (i >= 0) {
      Array.newInstance(Object.class, i);
    }
  }

  void testFor(Object a) {
    for (int i = 0; i < Array.getLength(a); ++i) {
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    }
  }

  void testMinLen(Object @MinLen(1) [] a) {
    Array.get(a, 0);
    // :: error: (argument)
    Array.get(a, 1);
      protected static int __cfwr_func744(Float __cfwr_p0, long __cfwr_p1, float __cfwr_p2) {
        return null;
        try {
            while (false) {
            if (false && ((true / 'x') + (353 % -459L))) {
            if (true || ('J' + (129L + true))) {
            while (((30.92 * 37) % null)) {
            try {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            return (false % -909);
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            try {
            while ((-25.73f - -16.81f)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        Integer __cfwr_entry47 = null;
        return 335;
    }
    protected Object __cfwr_handle32(Long __cfwr_p0) {
        for (int __cfwr_i20 = 0; __cfwr_i20 < 10; __cfwr_i20++) {
            float __cfwr_entry28 = 2.16f;
        }
        return null;
    }
}
