/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private static @IndexOrLow("#1") int lineStartIndexPartial(
      String s, @GTENegativeOne int lineStart) {
        try {
            short __cfwr_result99 = null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }

    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf("#1") int lineStartIndexPartial2(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    // :: error: (return)
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf("#1") int lineStartIndex2(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndex3(String s, int start) {
    if (s.length() == 0) {
      // :: error: (return)
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      // :: error: (return)
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      // :: error: (return)
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndex4(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
      public static Integer __cfwr_func648(byte __cfwr_p0, String __cfwr_p1) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        while (false) {
            while (((null & -53.55f) - 371)) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 5; __cfwr_i41++) {
            Character __cfwr_var88 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i95 = 0; __cfwr_i95 < 4; __cfwr_i95++) {
            return null;
        }
        return null;
    }
    protected static long __cfwr_handle415() {
        try {
            if (false || ((null >> 'W') >> null)) {
            return null;
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        return (74.03f | 394L);
        return -442L;
    }
    protected static Object __cfwr_aux298(Object __cfwr_p0, char __cfwr_p1) {
        for (int __cfwr_i76 = 0; __cfwr_i76 < 10; __cfwr_i76++) {
            boolean __cfwr_temp77 = true;
        }
        if (false && (true % ('z' & 72.84f))) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 1; __cfwr_i62++) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 9; __cfwr_i82++) {
            double __cfwr_temp55 = 62.07;
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 4; __cfwr_i69++) {
            if (true || true) {
            boolean __cfwr_val42 = false;
        }
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        return null;
    }
}
