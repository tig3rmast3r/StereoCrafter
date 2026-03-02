local mp = require("mp")
local msg = require("mp.msg")
local options = require("mp.options")

local opts = {
    csv_path = "/home/tig3mast3r/gitfork/StereoCrafter/sbs_click_annotations.csv",
    source_mode = "right_half",
    show_osd = "yes",
}

options.read_options(opts, "sbs_click_logger")

local function csv_escape(value)
    local s = tostring(value or "")
    s = s:gsub('"', '""')
    return '"' .. s .. '"'
end

local function basename(path)
    if not path then
        return ""
    end
    return path:match("([^/\\]+)$") or path
end

local function clamp(value, min_value, max_value)
    if value < min_value then
        return min_value
    end
    if value > max_value then
        return max_value
    end
    return value
end

local function ensure_header()
    local has_content = false
    local existing = io.open(opts.csv_path, "r")
    if existing then
        has_content = existing:read(1) ~= nil
        existing:close()
    end
    if has_content then
        return true
    end

    local f, err = io.open(opts.csv_path, "a")
    if not f then
        msg.error("unable to open CSV for header: " .. tostring(err))
        return false
    end
    f:write("timestamp,file_name,file_path,frame,time_sec,x_abs,y_abs,x_view,y_view,source_w,source_h,view_w,view_h\n")
    f:close()
    return true
end

local function get_view_coords()
    local mouse = mp.get_property_native("mouse-pos")
    local osd = mp.get_property_native("osd-dimensions")
    local out = mp.get_property_native("video-out-params")

    if not mouse or not osd or not out then
        return nil, "missing-properties"
    end

    local mx = tonumber(mouse.x)
    local my = tonumber(mouse.y)
    local ml = tonumber(osd.ml or 0)
    local mr = tonumber(osd.mr or 0)
    local mt = tonumber(osd.mt or 0)
    local mb = tonumber(osd.mb or 0)
    local osd_w = tonumber(osd.w or 0)
    local osd_h = tonumber(osd.h or 0)
    local out_w = tonumber(out.w or 0)
    local out_h = tonumber(out.h or 0)

    if not mx or not my then
        return nil, "mouse-unavailable"
    end

    local display_w = osd_w - ml - mr
    local display_h = osd_h - mt - mb
    if display_w <= 0 or display_h <= 0 or out_w <= 0 or out_h <= 0 then
        return nil, "invalid-display-area"
    end

    local x = (mx - ml) * out_w / display_w
    local y = (my - mt) * out_h / display_h

    local x_clamped = clamp(x, 0, math.max(out_w - 1, 0))
    local y_clamped = clamp(y, 0, math.max(out_h - 1, 0))

    return {
        x = math.floor(x_clamped + 0.5),
        y = math.floor(y_clamped + 0.5),
    }, nil
end

local function compute_abs_x(x_view, source_w, view_w)
    if opts.source_mode == "right_half" and source_w > view_w then
        return x_view + (source_w - view_w)
    end
    return x_view
end

local function append_click_row()
    if not ensure_header() then
        if opts.show_osd == "yes" then
            mp.osd_message("CSV non accessibile", 1.2)
        end
        return
    end

    local coords, coord_err = get_view_coords()
    if not coords then
        msg.warn("click ignored: " .. tostring(coord_err))
        if opts.show_osd == "yes" then
            mp.osd_message("Click non valido (" .. tostring(coord_err) .. ")", 1.0)
        end
        return
    end

    local video_src = mp.get_property_native("video-params") or {}
    local video_out = mp.get_property_native("video-out-params") or {}
    local source_w = tonumber(video_src.w or video_out.w or 0)
    local source_h = tonumber(video_src.h or video_out.h or 0)
    local view_w = tonumber(video_out.w or 0)
    local view_h = tonumber(video_out.h or 0)

    local x_abs = compute_abs_x(coords.x, source_w, view_w)
    local y_abs = coords.y

    local path = mp.get_property("path") or ""
    local frame = mp.get_property_number("estimated-frame-number", -1)
    local time_pos = mp.get_property_number("time-pos", -1)

    local row = table.concat({
        csv_escape(os.date("%Y-%m-%d %H:%M:%S")),
        csv_escape(basename(path)),
        csv_escape(path),
        tostring(math.floor(frame + 0.5)),
        string.format("%.6f", time_pos),
        tostring(math.floor(x_abs)),
        tostring(math.floor(y_abs)),
        tostring(math.floor(coords.x)),
        tostring(math.floor(coords.y)),
        tostring(math.floor(source_w)),
        tostring(math.floor(source_h)),
        tostring(math.floor(view_w)),
        tostring(math.floor(view_h)),
    }, ",") .. "\n"

    local f, open_err = io.open(opts.csv_path, "a")
    if not f then
        msg.error("unable to append CSV: " .. tostring(open_err))
        if opts.show_osd == "yes" then
            mp.osd_message("Errore apertura CSV", 1.2)
        end
        return
    end

    local ok, write_err = f:write(row)
    f:close()

    if not ok then
        msg.error("unable to write CSV row: " .. tostring(write_err))
        if opts.show_osd == "yes" then
            mp.osd_message("Errore scrittura CSV", 1.2)
        end
        return
    end

    if opts.show_osd == "yes" then
        mp.osd_message(
            string.format(
                "CSV append: %s f=%d x=%d y=%d",
                basename(path),
                math.floor(frame + 0.5),
                math.floor(x_abs),
                math.floor(y_abs)
            ),
            0.8
        )
    end
end

local function on_left_click(event)
    if event.event == "down" then
        append_click_row()
    end
end

mp.register_event("file-loaded", ensure_header)
mp.add_forced_key_binding("MBTN_LEFT", "sbs_click_logger_left_click", on_left_click, { complex = true })
msg.info("sbs_click_logger loaded; CSV: " .. opts.csv_path)
